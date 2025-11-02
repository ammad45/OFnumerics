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

#include "hybridIterGaussLS.H"
#include "gaussGrad.H"
#include "leastSquaresGrad.H"
#include "iterativeGaussGrad.H"
#include "extrapolatedCalculatedFvPatchField.H"
#include "IStringStream.H"
#include "zeroGradientFvPatchFields.H"
#include "objectRegistry.H"

// * * * * * * * * * * * * * Private helpers * * * * * * * * * * * * * * * * //

template<class Type>
void Foam::fv::hybridIterGaussLS<Type>::computeCellQuality(Foam::scalarField& metric) const
{
    const fvMesh& mesh = this->mesh();

    const vectorField& C  = mesh.C();
    const vectorField& Cf = mesh.Cf();
    const vectorField& Sf = mesh.Sf();

    const labelUList& owner    = mesh.owner();
    const labelUList& neighbour= mesh.neighbour();
    const label nInternalFaces = mesh.nInternalFaces();

    metric.setSize(mesh.nCells());

    const auto& cells = mesh.cells();

    forAll(metric, cellI)
    {
        scalar maxNonOrtho = 0;
        scalar maxSkew     = 0;
        const labelList& cFaces = cells[cellI];
        const vector& CP = C[cellI];

        forAll(cFaces, i)
        {
            const label fI = cFaces[i];

            const vector dcf = Cf[fI] - CP;
            const scalar dMag = mag(dcf);
            const vector& S = Sf[fI];

            if (dMag > SMALL && mag(S) > SMALL)
            {
                const scalar co = (dcf & S)/(dMag*mag(S));
                maxNonOrtho = max(maxNonOrtho, 1 - sqr(co));
            }

            if (fI < nInternalFaces)
            {
                const label own = owner[fI];
                const label nei = neighbour[fI];
                const vector AB = C[nei] - C[own];
                const scalar ABmag = mag(AB);
                if (ABmag > SMALL)
                {
                    const scalar dSkew = mag((Cf[fI] - C[own]) ^ AB)/(ABmag + SMALL);
                    const scalar skew  = min(dSkew/(ABmag + SMALL)*2.0, scalar(1));
                    maxSkew = max(maxSkew, skew);
                }
            }
        }

        metric[cellI] = min(max(max(maxNonOrtho, maxSkew), scalar(0)), scalar(1));
    }
}


template<class Type>
bool Foam::fv::hybridIterGaussLS<Type>::computeMetricFromStability
(
    const word& fieldName,
    Foam::scalarField& metric
) const
{
    const fvMesh& mesh = this->mesh();
    const word faceName(fieldName + "BlendingFactor");
    const surfaceScalarField* bfPtr = mesh.findObject<surfaceScalarField>(faceName);
    if (!bfPtr) return false;

    const surfaceScalarField& bf = *bfPtr;

    metric.setSize(mesh.nCells());
    metric = 0;

    const labelUList& owner    = mesh.owner();
    const labelUList& neighbour= mesh.neighbour();

    // Accumulate face values to cells
    labelList counts(mesh.nCells(), 0);

    forAll(owner, facei)
    {
        const scalar v = bf[facei];
        metric[owner[facei]]    += v;
        metric[neighbour[facei]]+= v;
        ++counts[owner[facei]];
        ++counts[neighbour[facei]];
    }

    // Boundary faces: count towards owner only
    forAll(mesh.boundary(), patchi)
    {
        const auto& p = mesh.boundary()[patchi];
        const labelUList& pFaceCells = p.faceCells();
        const auto& pbf = bf.boundaryField()[patchi];
        forAll(p, i)
        {
            metric[pFaceCells[i]] += pbf[i];
            ++counts[pFaceCells[i]];
        }
    }

    forAll(metric, cellI)
    {
        const label n = max(counts[cellI], 1);
        // Interpret face factor as a "badness" proxy: higher -> worse
        metric[cellI] = metric[cellI]/n;
        // Clamp to [0,1]
        metric[cellI] = min(max(metric[cellI], scalar(0)), scalar(1));
    }

    return true;
}


// * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * * * * //

template<class Type>
Foam::fv::hybridIterGaussLS<Type>::hybridIterGaussLS
(
    const fvMesh& mesh,
    Istream& schemeData
)
:
    gradScheme<Type>(mesh),
    tinterpScheme_(nullptr),
    goodThresh_(0.15),
    badThresh_(0.60)
{
    if (schemeData.eof())
    {
        tinterpScheme_.reset(new linear<Type>(mesh));
    }
    else
    {
        tinterpScheme_.reset(surfaceInterpolationScheme<Type>::New(mesh, schemeData));
    }

    // Optional: thresholds
    if (!schemeData.eof()) goodThresh_ = readScalar(schemeData);
    if (!schemeData.eof()) badThresh_  = readScalar(schemeData);

    if (goodThresh_ < 0 || goodThresh_ >= badThresh_ || badThresh_ > 1)
    {
        WarningInFunction
            << "Adjusting invalid thresholds: goodThresh=" << goodThresh_
            << " badThresh=" << badThresh_ << nl;
        goodThresh_ = 0.15;
        badThresh_  = 0.60;
    }

    // iterative Gauss iterations fixed to 2 in calcGrad
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
Foam::fv::hybridIterGaussLS<Type>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{
    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    const fvMesh& mesh = this->mesh();

    // Decide metric source: external stability field or internal quality
    scalarField metric;
    if (!computeMetricFromStability(vsf.name(), metric))
    {
        computeCellQuality(metric);
    }

    // Fast gating: if all cells are clearly good or bad, compute one scheme only
    {
        scalar minM = GREAT;
        scalar maxM = -GREAT;
        forAll(metric, i)
        {
            const scalar mi = metric[i];
            if (mi < minM) minM = mi;
            if (mi > maxM) maxM = mi;
        }

        if (maxM <= goodThresh_ + SMALL)
        {
            // All cells good -> iterative Gauss only (2 iterations)
            typedef typename outerProduct<vector, Type>::type GradType;
            typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

            const word interpName = tinterpScheme_().type();
            IStringStream iterData(Foam::string(interpName) + " " + Foam::name(2));
            fv::iterativeGaussGrad<Type> iterG(mesh, iterData);
            tmp<GradFieldType> tItOnly = iterG.calcGrad(vsf, "iterativeGaussGrad(" + vsf.name() + ")");

            tmp<GradFieldType> tRet
            (
                new GradFieldType
                (
                    IOobject(name, vsf.instance(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                    tItOnly()
                )
            );
            return tRet;
        }

        if (minM >= badThresh_ - SMALL)
        {
            // All cells bad -> least-squares only
            typedef typename outerProduct<vector, Type>::type GradType;
            typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

            tmp<GradFieldType> tLsOnly = fv::leastSquaresGrad<Type>(mesh).calcGrad
            (
                vsf, "leastSquaresGrad(" + vsf.name() + ")"
            );

            tmp<GradFieldType> tRet
            (
                new GradFieldType
                (
                    IOobject(name, vsf.instance(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                    tLsOnly()
                )
            );
            return tRet;
        }
    }

    // Export metric for visualization (0..1) as a volScalarField
    {
        const word mName("hybridIterGaussLSMetric_" + vsf.name());
        objectRegistry& obr = const_cast<objectRegistry&>(mesh.thisDb());
        if (auto* mPtr = obr.getObjectPtr<volScalarField>(mName))
        {
            mPtr->primitiveFieldRef() = metric;
        }
        else
        {
            auto* newM = new volScalarField
            (
                IOobject
                (
                    mName,
                    vsf.instance(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE,
                    IOobject::REGISTER
                ),
                mesh,
                dimensionedScalar(dimless, Zero),
                fvPatchFieldBase::zeroGradientType()
            );
            newM->primitiveFieldRef() = metric;
            regIOobject::store(newM);
        }
    }

    // Prepare result
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

    // Compute iterative Gauss once (needed for boundaries / blending)
    tmp<GradFieldType> tIt;
    {
        const word interpName = tinterpScheme_().type();
        IStringStream iterData(Foam::string(interpName) + " " + Foam::name(2));
        fv::iterativeGaussGrad<Type> iterG(mesh, iterData);
        tIt = iterG.calcGrad(vsf, "iterativeGaussGrad(" + vsf.name() + ")");
    }

    // Compute Least-Squares once
    tmp<GradFieldType> tLs = fv::leastSquaresGrad<Type>(mesh).calcGrad
    (
        vsf, "leastSquaresGrad(" + vsf.name() + ")"
    );
    const GradFieldType& gLS = tLs();

    // Reference to iterative Gauss
    const GradFieldType& gIt = tIt();

    // Single pass over cells: smoothstep between Iterative (good) and LS (bad)
    forAll(g, cellI)
    {
        const scalar m = min(max(metric[cellI], scalar(0)), scalar(1));

        scalar r = 0;
        if (badThresh_ > goodThresh_)
        {
            r = (m - goodThresh_)/max(badThresh_ - goodThresh_, VSMALL);
        }
        r = min(max(r, scalar(0)), scalar(1));

        // Smoothstep s(r) = r^2*(3 - 2r)
        const scalar s = r*r*(3 - 2*r);

        // good -> iterativeGauss, bad -> leastSquares
        g[cellI] = (1.0 - s)*gIt[cellI] + s*gLS[cellI];
    }

    // Boundaries: iterative Gauss-only
    {
        auto& gbf = g.boundaryFieldRef();
        const auto& gitbf = gIt.boundaryField();
        forAll(gbf, patchi)
        {
            gbf[patchi] = gitbf[patchi];
        }
    }

    // Final boundary correction consistent with Gauss
    fv::gaussGrad<Type>::correctBoundaryConditions(vsf, g);

    return tGrad;
}


// ************************************************************************* //
