/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Class
        Foam::fv::blendedGradScheme

    Group
        grpFvGradSchemes

    Description
        Gradient scheme that smoothly blends two gradient schemes based on
        a precomputed volScalarField (gradSchemeBlending, 0..1).

        The blending factor β is read from the field:
        - β = 0: use background scheme (default: leastSquares)
        - β = 1: use blended scheme (default: gauss)
        - 0 < β < 1: smooth linear blend

        The final gradient is:
            ∇φ = (1-β) ∇φ_background + β ∇φ_blended

        This scheme is efficient because blending factors are precomputed
        once before the simulation, avoiding repeated quality calculations.

    Usage
        In fvSchemes:
            gradSchemes
            {
                default blendedGradScheme leastSquares gauss;
                // or
                default blendedGradScheme <backgroundScheme> <blendedScheme>;
            }

        The blending field must be precomputed using computeGradSchemeBlending.

    SourceFiles
        blendedGradScheme.C

\*---------------------------------------------------------------------------*/

#ifndef blendedGradScheme_H
#define blendedGradScheme_H

#include "gradScheme.H"

namespace Foam
{
namespace fv
{

template<class Type>
class blendedGradScheme
:
    public fv::gradScheme<Type>
{
    // Private Data

        //- Background scheme (used when β=0)
        tmp<fv::gradScheme<Type>> backgroundScheme_;

        //- Blended scheme (used when β=1)
        tmp<fv::gradScheme<Type>> blendedScheme_;

        //- Original textual specs (for logging)
        string backgroundSpec_;
        string blendedSpec_;

        //- Name of blending field (default: gradSchemeBlending)
        word blendingFieldName_;

    // Private Member Functions

        //- No copy construct
        blendedGradScheme(const blendedGradScheme&) = delete;

        //- No copy assignment
        void operator=(const blendedGradScheme&) = delete;

public:

    //- Runtime type information
    TypeName("blendedGradScheme");

    // Constructors

        //- Construct from mesh
        blendedGradScheme(const fvMesh& mesh)
        :
            gradScheme<Type>(mesh),
            blendingFieldName_("gradSchemeBlending")
        {}

        //- Construct from mesh and Istream
        //  Format:
        //    blendedGradScheme
        //    {
        //        scheme1 <any gradScheme spec, arbitrary tokens>;
        //        scheme2 <any gradScheme spec, arbitrary tokens>;
        //    }
        //  or positional fallback:
        //    blendedGradScheme <scheme1> <scheme2...>
        //  (first token is scheme1; all remaining tokens compose scheme2)
        blendedGradScheme(const fvMesh& mesh, Istream& schemeData);

    // Member Functions

        //- Calculate gradient
        virtual tmp< GeometricField< typename outerProduct<vector, Type>::type, fvPatchField, volMesh> > calcGrad
        (
            const GeometricField<Type, fvPatchField, volMesh>& vsf,
            const word& name
        ) const;
};

} // End namespace fv
} // End namespace Foam

#ifdef NoRepository
    #include "blendedGradScheme.C"
#endif

#endif

// ************************************************************************* //

