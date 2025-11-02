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

Class
    Foam::fv::hybridIterGaussLS

Group
    grpFvGradSchemes

Description
    Hybrid gradient scheme blending iterativeGauss (2 iters) and least-squares
    using a simple, smooth metric-based transition:

    - Normal (good) cells (metric <= goodThresh): iterative Gauss (2 iterations)
    - Poor (bad) cells   (metric >= badThresh): least-squares
    - Between good/bad: smoothstep blend from iterative Gauss -> least-squares

    The per-cell metric is taken from a face-based stability blending field
    <field>BlendingFactor if present (converted to cell values). Otherwise it
    falls back to an internal proxy metric (max of non-orthogonality, skewness).

    Boundary patches use iterative Gauss only.

    Usage example in system/fvSchemes:

        gradSchemes
        {
            grad(U)    hybridIterGaussLS linear 0.15 0.6; // interp good bad
        }

    Parameters (after the interpolation scheme):
      - goodThresh (scalar, default 0.15)
      - badThresh  (scalar, default 0.60)
      - iterative Gauss uses 2 iterations (fixed)

SourceFiles
    hybridIterGaussLS.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_hybridIterGaussLS_H
#define Foam_hybridIterGaussLS_H

#include "gradScheme.H"
#include "surfaceInterpolationScheme.H"
#include "linear.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace fv
{

/*---------------------------------------------------------------------------*\
                       Class hybridIterGaussLS Declaration
\*---------------------------------------------------------------------------*/

template<class Type>
class hybridIterGaussLS
:
    public fv::gradScheme<Type>
{
    // Private Data

        //- Interpolation scheme used by iterative Gauss
        tmp<surfaceInterpolationScheme<Type>> tinterpScheme_;

        //- Thresholds and parameters
        scalar goodThresh_;
        scalar badThresh_;


    // Private Member Functions

        //- No copy construct
        hybridIterGaussLS(const hybridIterGaussLS&) = delete;

        //- No copy assignment
        void operator=(const hybridIterGaussLS&) = delete;

        //- Compute internal cell quality metric (max(non-ortho, skew))
        void computeCellQuality(scalarField& metric) const;

        //- Try to compute metric from face-based <field>BlendingFactor
        //  Returns true if used, false otherwise
        bool computeMetricFromStability
        (
            const word& fieldName,
            scalarField& metric
        ) const;


public:

    //- Runtime type information
    TypeName("hybridIterGaussLS");


    // Constructors

        //- Construct from mesh (defaults)
        hybridIterGaussLS(const fvMesh& mesh)
        :
            gradScheme<Type>(mesh),
            tinterpScheme_(new linear<Type>(mesh)),
            goodThresh_(0.15),
            badThresh_(0.60)
        {}

        //- Construct from mesh and Istream (interp + thresholds)
        hybridIterGaussLS(const fvMesh& mesh, Istream& schemeData);


    // Member Functions

        //- Return the gradient of the given field to the gradScheme::grad
        //- for optional caching
        virtual tmp
        <
            GeometricField
            <typename outerProduct<vector, Type>::type, fvPatchField, volMesh>
        > calcGrad
        (
            const GeometricField<Type, fvPatchField, volMesh>& vsf,
            const word& name
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace fv

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "hybridIterGaussLS.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
