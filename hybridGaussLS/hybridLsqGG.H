/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Class
        Foam::fv::hybridLsqGG

    Group
        grpFvGradSchemes

    Description
        Hybrid Gauss–Least-Squares gradient scheme as implemented in commercial codes
        such as Ansys Fluent and Star-CCM+  . The cell
        blending factor β is a quality criterion computed from:
          - LSQ tensor quality (eigenvalue ratio)
          - Flat cell curvature criterion (tan(skewAngle) vs NCF*aspect)
          - Cell skewness criterion (piecewise-linear between safe/unsafe)
          - Chevron-cell criterion (β=0 if chevron)
        with β = min of the above. The final gradient is
            ∇φ = β ∇φ_LSQ + (1-β) ∇φ_GG
        Optional gradient limiting using Venkatakrishnan or modified
        Venkatakrishnan limiters, applied component-wise for vector fields.

        Boundaries: Green–Gauss.

    SourceFiles
        hybridLsqGG.C

\*---------------------------------------------------------------------------*/

#ifndef hybridLsqGG_H
#define hybridLsqGG_H

#include "gradScheme.H"
#include "surfaceInterpolationScheme.H"
#include "linear.H"

namespace Foam
{
namespace fv
{

template<class Type>
class hybridLsqGG
:
    public fv::gradScheme<Type>
{
    // Private Data

        //- Interpolation scheme used by Green–Gauss
        tmp<surfaceInterpolationScheme<Type>> tinterpScheme_;

        //- Per-helper activation flags (opt-in)
        bool useChevron_;
        //- Aspect ratio threshold helper (simple flat-cells criterion)
        bool useAspect_;
        scalar aspectThresh_;

        //- Flat curvature helper: tan(skewAngle) vs NCF*aspect, with linear decay
        bool useFlat_;
        scalar flatNCF_;
        scalar flatDecay_;

        //- LSQ eigenvalue ratio helper (stencil conditioning)
        bool useLsqRatio_;
        scalar lsqEigenRatioMin_;

        //- Limiter selection
        enum limiterType
        {
            limNone,
            limVenkat,
            limVenkatMod
        } limiter_;

        //- Enable/disable gradient limiting (optional)
        bool enableLimiter_;
        // (no base method toggle; LSQ is default, Gauss used only where helpers flag beta=0)

    // Private Member Functions

        //- No copy construct
        hybridLsqGG(const hybridLsqGG&) = delete;

        //- No copy assignment
        void operator=(const hybridLsqGG&) = delete;

        //- Compute β for all cells (0..1)
        void computeBeta(scalarField& beta) const;

        // Criteria helpers (return in 0..1)
        scalar betaChevron(const label cellI) const;

        //- Aspect ratio proxy per cell
        scalar aspectRatio(const label cellI) const;

        //- Max skew angle (deg) over faces of cell (cheap proxy)
        scalar maxSkewAngleDeg(const label cellI) const;

        //- Flat curvature beta
        scalar betaFlatCurvature(const label cellI) const;

        //- LSQ quality beta
        scalar betaLsqQuality(const label cellI) const;

        //- Apply limiter after forming unlimited gradient (component-wise for vectors)
        void applyLimiter(
            const GeometricField<Type, fvPatchField, volMesh>& vsf,
            GeometricField<typename outerProduct<vector, Type>::type, fvPatchField, volMesh>& g
        ) const;

        //- Scalar limiter core (computes α per cell and scales g)
        void limitScalarGradient(const volScalarField& vsf, volVectorField& g) const;

        //- Vector limiter core: component-wise
        void limitVectorGradient(const volVectorField& vsf, volTensorField& g) const;

public:

    //- Runtime type information
    TypeName("hybridLsqGG");

    // Constructors

        //- Construct from mesh
        hybridLsqGG(const fvMesh& mesh)
        :
            gradScheme<Type>(mesh),
            tinterpScheme_(new linear<Type>(mesh)),
            useChevron_(false),
            useAspect_(false),
            aspectThresh_(10.0),
            useFlat_(false),
            flatNCF_(0.2),
            flatDecay_(2.0),
            useLsqRatio_(false),
            lsqEigenRatioMin_(2.0),
            limiter_(limNone),
            enableLimiter_(false)
        {}

        //- Construct from mesh and Istream (interp + thresholds + limiter)
        hybridLsqGG(const fvMesh& mesh, Istream& schemeData);

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
    #include "hybridLsqGG.C"
#endif

#endif

// ************************************************************************* //
