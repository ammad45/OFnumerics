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
    Foam::fv::hybridGaussLS

Group
    grpFvGradSchemes

Description
    Gradient scheme that blends Gauss (Green–Gauss) and least-squares (LS)
    using a per-cell quality metric. The blending weight is computed from
    a combination of non-orthogonality and skewness proxies (max of both).

    - Internal cells: blended Gauss/LS per metric.
    - Boundary patches: Gauss-only gradient (no blending).

    Usage example in system/fvSchemes:

        gradSchemes
        {
            grad(U)    hybridGaussLS linear 0.5 0.3; // interp alpha thresh
        }

    Parameters (after the interpolation scheme):
      - alpha (scalar, default 0.5): global weight scaling for LS
      - alphaSkewThresh (scalar, default 0.3): metric threshold for ramp

SourceFiles
    hybridGaussLS.C

\*---------------------------------------------------------------------------*/

#ifndef Foam_hybridGaussLS_H
#define Foam_hybridGaussLS_H

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
                       Class hybridGaussLS Declaration
\*---------------------------------------------------------------------------*/

template<class Type>
class hybridGaussLS
:
    public fv::gradScheme<Type>
{
    // Private Data

        //- Interpolation scheme used for the Gauss part
        tmp<surfaceInterpolationScheme<Type>> tinterpScheme_;

        //- Blending parameters
        scalar alphaGlobal_;
        scalar alphaSkewThresh_;


    // Private Member Functions

        //- No copy construct
        hybridGaussLS(const hybridGaussLS&) = delete;

        //- No copy assignment
        void operator=(const hybridGaussLS&) = delete;

        //- Compute a simple cell quality metric in [0,1]
        void computeCellQuality(scalarField& metric) const;

        //- Map metric -> LS blending weight
        inline scalar blendAlpha(const scalar m) const;


public:

    //- Runtime type information
    TypeName("hybridGaussLS");


    // Constructors

        //- Construct from mesh (defaults: linear interpolation, defaults params)
        hybridGaussLS(const fvMesh& mesh)
        :
            gradScheme<Type>(mesh),
            tinterpScheme_(new linear<Type>(mesh)),
            alphaGlobal_(0.5),
            alphaSkewThresh_(0.3)
        {}

        //- Construct from mesh and Istream (interp + optional params)
        hybridGaussLS(const fvMesh& mesh, Istream& schemeData);


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
    #include "hybridGaussLS.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
