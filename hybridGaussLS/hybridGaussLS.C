/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2025
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "hybridGaussLS.H"
#include "gaussGrad.H"
#include "leastSquaresGrad.H"
#include "extrapolatedCalculatedFvPatchField.H"

// * * * * * * * * * * * * * Private helpers * * * * * * * * * * * * * * * * //

template<class Type>
void Foam::fv::hybridGaussLS<Type>::computeCellQuality(Foam::scalarField& metric) const
{
    const fvMesh& mesh = this->mesh();

    const vectorField& C = mesh.C();
    const vectorField& Cf = mesh.Cf();
    const vectorField& Sf = mesh.Sf();

    const label nCells = mesh.nCells();
    metric.setSize(nCells);

    const auto& cells = mesh.cells();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const label nInternalFaces = mesh.nInternalFaces();

    forAll(metric, cellI)
    {
        scalar maxNonOrtho = 0;
        scalar maxSkew = 0;
        const labelList& cFaces = cells[cellI];
        const vector& CP = C[cellI];

        forAll(cFaces, i)
        {
            const label fI = cFaces[i];

            // Non-orthogonality proxy based on centre-to-face vector vs face normal
            const vector dcf = Cf[fI] - CP;
            const scalar dMag = mag(dcf);
            const vector& S = Sf[fI];

            if (dMag > SMALL && mag(S) > SMALL)
            {
                const scalar co = (dcf & S)/(dMag*mag(S));
                const scalar nonOrtho = 1 - sqr(co); // 0 good, ->1 bad
                maxNonOrtho = max(maxNonOrtho, nonOrtho);
            }

            // Skewness proxy: distance of Cf to the owner-neighbour line (internal faces only)
            if (fI < nInternalFaces)
            {
                const label own = owner[fI];
                const label nei = neighbour[fI];
                const vector AB = C[nei] - C[own];
                const scalar ABmag = mag(AB);
                if (ABmag > SMALL)
                {
                    // distance to line passing through owner with direction AB
                    const scalar dSkew = mag((Cf[fI] - C[own]) ^ AB)/(ABmag + SMALL);
                    // normalize by AB length to yield dimensionless measure in [0,1] approx
                    const scalar skew = min(dSkew/(ABmag + SMALL)*2.0, scalar(1));
                    maxSkew = max(maxSkew, skew);
                }
            }
        }

        // Combine metrics (conservative): take the maximum
        const scalar m = max(maxNonOrtho, maxSkew);
        metric[cellI] = min(max(m, scalar(0)), scalar(1));
    }
}


template<class Type>
inline Foam::scalar Foam::fv::hybridGaussLS<Type>::blendAlpha(const Foam::scalar m) const
{
    if (m <= alphaSkewThresh_) return alphaGlobal_*(m/alphaSkewThresh_);

    const scalar res = alphaGlobal_ + (1.0 - alphaGlobal_)
        *((m - alphaSkewThresh_)/(max(1.0 - alphaSkewThresh_, VSMALL)));

    return min(max(res, scalar(0)), scalar(1));
}


// * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * * * * //

template<class Type>
Foam::fv::hybridGaussLS<Type>::hybridGaussLS(const fvMesh& mesh, Istream& schemeData)
:
    gradScheme<Type>(mesh),
    tinterpScheme_(nullptr),
    alphaGlobal_(0.5),
    alphaSkewThresh_(0.3)
{
    // Parse interpolation scheme first (or default to linear)
    if (schemeData.eof())
    {
        tinterpScheme_.reset(new linear<Type>(mesh));
    }
    else
    {
        tinterpScheme_.reset(surfaceInterpolationScheme<Type>::New(mesh, schemeData));
    }

    // Optional parameters (alpha, alphaSkewThresh)
    if (!schemeData.eof()) alphaGlobal_ = readScalar(schemeData);
    if (!schemeData.eof()) alphaSkewThresh_ = readScalar(schemeData);
}


// * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * * //

template<class Type>
Foam::tmp
<
    Foam::GeometricField
    <
        typename Foam::outerProduct<Foam::vector, Type>::type,
        Foam::fvPatchField,
        Foam::volMesh
    >
>
Foam::fv::hybridGaussLS<Type>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{
    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    const fvMesh& mesh = this->mesh();

    // Gauss gradient using chosen interpolation
    tmp<GradFieldType> tGg = fv::gaussGrad<Type>(mesh).calcGrad(vsf, "GaussGrad(" + vsf.name() + ")");
    const GradFieldType& gg = tGg();

    // Least-squares gradient (second order)
    tmp<GradFieldType> tLs = fv::leastSquaresGrad<Type>(mesh).calcGrad(vsf, "leastSquaresGrad(" + vsf.name() + ")");
    const GradFieldType& ls = tLs();

    // Compute metric
    scalarField metric;
    computeCellQuality(metric);

    // Allocate result
    tmp<GradFieldType> tGrad
    (
        new GradFieldType
        (
            IOobject
            (
                name,
                vsf.instance(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensioned<GradType>(vsf.dimensions()/dimLength, Zero),
            fvPatchFieldBase::extrapolatedCalculatedType()
        )
    );
    GradFieldType& g = tGrad.ref();

    // Blend per cell
    forAll(g, cellI)
    {
        const scalar a = blendAlpha(metric[cellI]);
        g[cellI] = (1.0 - a)*gg[cellI] + a*ls[cellI];
    }

    // On boundaries: use Gauss gradient (no blending)
    {
        auto& gbf = g.boundaryFieldRef();
        const auto& ggbf = gg.boundaryField();
        forAll(gbf, patchi)
        {
            gbf[patchi] = ggbf[patchi];
        }
    }

    // Boundary: use Gauss correction logic for consistency
    fv::gaussGrad<Type>::correctBoundaryConditions(vsf, g);

    return tGrad;
}


// ************************************************************************* //
