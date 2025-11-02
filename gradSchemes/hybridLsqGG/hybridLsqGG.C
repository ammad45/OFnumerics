#include "hybridLsqGG.H"
#include "gaussGrad.H"
#include "leastSquaresGrad.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "IStringStream.H"
#include "mathematicalConstants.H"

namespace Foam
{
namespace fv
{

// ---- ctor with parameters ----

template<class Type>
hybridLsqGG<Type>::hybridLsqGG(const fvMesh& mesh, Istream& schemeData)
:
    gradScheme<Type>(mesh),
    tinterpScheme_(nullptr),
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
{
    if (schemeData.eof())
    {
        tinterpScheme_.reset(new linear<Type>(mesh));
        return;
    }

    // Interpolation scheme for GG
    tinterpScheme_.reset(surfaceInterpolationScheme<Type>::New(mesh, schemeData));

    // Optional parameters and toggles parsed as words:
    //   aspect <threshold>
    //   chevron on|off
    //   limiter venkat|venkatMod|none
    while (!schemeData.eof())
    {
        word key; schemeData >> key;
        if (!schemeData.good()) break;

        if (key == "aspect")
        {
            aspectThresh_ = readScalar(schemeData);
            useAspect_ = true;
        }
        else if (key == "chevron")
        {
            word v; schemeData >> v; useChevron_ = (v == "on" || v == "true" || v == "1");
        }
        else if (key == "flat")
        {
            flatNCF_ = readScalar(schemeData);
            if (!schemeData.eof())
            {
                token t(schemeData);
                if (t.isNumber()) flatDecay_ = scalar(t.number()); else schemeData.putBack(t);
            }
            useFlat_ = true;
        }
        else if (key == "lsq")
        {
            lsqEigenRatioMin_ = readScalar(schemeData);
            useLsqRatio_ = true;
        }
        else if (key == "limiter")
        {
            word v; schemeData >> v;
            if (v == "venkat") { limiter_ = limVenkat; enableLimiter_ = true; }
            else if (v == "venkatMod") { limiter_ = limVenkatMod; enableLimiter_ = true; }
            else { limiter_ = limNone; enableLimiter_ = false; }
        }
        else
        {
            // Unknown token, stop parsing to avoid consuming next dictionary
            break;
        }
    }
}

// ---- helpers ----

// Aspect ratio proxy: max neighbour distance / min neighbour distance
// (simple, robust proxy)

template<class Type>
scalar hybridLsqGG<Type>::aspectRatio(const label cellI) const
{
    const fvMesh& mesh = this->mesh();
    const vector& C0 = mesh.C()[cellI];
    const labelList& cf = mesh.cells()[cellI];
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const label nInt = mesh.nInternalFaces();

    scalar dmin = GREAT;
    scalar dmax = -GREAT;

    forAll(cf, i)
    {
        const label fI = cf[i];
        label other = -1;
        if (fI < nInt)
        {
            const label o = own[fI];
            const label n = nei[fI];
            other = (o == cellI) ? n : o;
        }
        else
        {
            // approximate with distance to face centre
            const scalar d = mag(mesh.Cf()[fI] - C0);
            dmin = min(dmin, d);
            dmax = max(dmax, d);
            continue;
        }
        if (other >= 0)
        {
            const scalar d = mag(mesh.C()[other] - C0);
            dmin = min(dmin, d);
            dmax = max(dmax, d);
        }
    }
    if (dmin <= SMALL) return 1.0;
    return max(dmax/dmin, scalar(1));
}

// Chevron-cell criterion from dp projection norm; β=0 if c<0 else 1

template<class Type>
scalar hybridLsqGG<Type>::betaChevron(const label cellI) const
{
    const fvMesh& mesh = this->mesh();
    const labelList& cf = mesh.cells()[cellI];
    const vector& C0 = mesh.C()[cellI];
    const vectorField& Cf = mesh.Cf();
    const vectorField& Sf = mesh.Sf();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const label nInt = mesh.nInternalFaces();

    forAll(cf, i)
    {
        const label fI = cf[i];
        vector ds(Zero);
        if (fI < nInt)
        {
            const label o = own[fI];
            const label n = nei[fI];
            const label other = (o == cellI) ? n : o;
            ds = mesh.C()[other] - C0;
        }
        else
        {
            ds = Cf[fI] - C0; // fallback
        }
        const vector dx = Cf[fI] - C0;
        const vector Af = Sf[fI];
        const scalar Af_dx = Af & dx;
        const scalar Af_ds = Af & ds;
        vector dp = ((Af_ds != 0) ? (Af_dx/Af_ds)*ds : vector::zero) - dx;

        // max(dp·dv) over face vertices
        scalar maxdpdv = SMALL;
        const face& f = mesh.faces()[fI];
        forAll(f, pti)
        {
            const point& pt = mesh.points()[f[pti]];
            const vector dv = Cf[fI] - vector(pt);
            maxdpdv = max(maxdpdv, dp & dv);
        }
        const scalar c = 1.0 - ((dp & dp)/maxdpdv);
        if (c < 0) return 0.0;
    }
    return 1.0;
}

// Maximum skewness angle over faces of cell (deg)

template<class Type>
scalar hybridLsqGG<Type>::maxSkewAngleDeg(const label cellI) const
{
    const fvMesh& mesh = this->mesh();
    const vectorField& C = mesh.C();
    const vectorField& Cf = mesh.Cf();
    const vectorField& Sf = mesh.Sf();
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const label nInt = mesh.nInternalFaces();

    const vector& C0 = C[cellI];
    scalar amax = 0;
    const labelList& cFaces = mesh.cells()[cellI];
    forAll(cFaces, i)
    {
        const label fI = cFaces[i];
        vector AB(Zero);
        if (fI < nInt)
        {
            const label o = own[fI];
            const label n = nei[fI];
            const label other = (o == cellI) ? n : o;
            AB = C[other] - C0;
        }
        else
        {
            AB = Sf[fI];
        }
        const vector dcf = Cf[fI] - C0;
        const scalar c = (mag(AB) > SMALL && mag(dcf) > SMALL) ? (dcf & AB)/(mag(dcf)*mag(AB)) : 1;
        const scalar ang = acos(max(min(c, scalar(1)), scalar(-1)))
                         * (180.0/Foam::constant::mathematical::pi);
        if (ang > amax) amax = ang;
    }
    return amax;
}

// Flat curvature beta

template<class Type>
scalar hybridLsqGG<Type>::betaFlatCurvature(const label cellI) const
{
    const scalar ang = maxSkewAngleDeg(cellI);
    const scalar t = tan(ang*Foam::constant::mathematical::pi/180.0);
    const scalar ar = aspectRatio(cellI);
    const scalar tau = flatNCF_*ar;
    if (t <= tau) return 1.0;
    const scalar t2 = flatDecay_*tau;
    if (t >= t2) return 0.0;
    return max(scalar(0), 1.0 - (t - tau)/max(t2 - tau, SMALL));
}

// LSQ quality beta (diagonal dd proxy)

template<class Type>
scalar hybridLsqGG<Type>::betaLsqQuality(const label cellI) const
{
    const fvMesh& mesh = this->mesh();
    const vectorField& C = mesh.C();
    const labelList& cf = mesh.cells()[cellI];
    symmTensor dd(Zero);
    const labelUList& own = mesh.owner();
    const labelUList& nei = mesh.neighbour();
    const label nInt = mesh.nInternalFaces();

    forAll(cf, i)
    {
        const label fI = cf[i];
        vector d(Zero);
        if (fI < nInt)
        {
            const label o = own[fI];
            const label n = nei[fI];
            const label other = (o == cellI) ? n : o;
            d = C[other] - C[cellI];
        }
        else
        {
            d = mesh.Cf()[fI] - C[cellI];
        }
        const scalar d2 = magSqr(d);
        if (d2 > VSMALL) dd += sqr(d)/d2;
    }
    const scalar dmin = max(min(dd.xx(), min(dd.yy(), dd.zz())), SMALL);
    const scalar dmax = max(dd.xx(), max(dd.yy(), dd.zz()));
    const scalar ratio = dmax/dmin;
    // High ratio => poorly conditioned LSQ stencil -> beta=0 (use Gauss)
    return (ratio >= lsqEigenRatioMin_) ? scalar(0) : scalar(1);
}

// Compute beta per cell

template<class Type>
void hybridLsqGG<Type>::computeBeta(scalarField& beta) const
{
    const fvMesh& mesh = this->mesh();
    beta.setSize(mesh.nCells());
    // Default all-LSQ
    forAll(beta, i) beta[i] = 1.0;

    // Aspect ratio helper → set beta=0 where aspect >= threshold
    if (useAspect_)
    {
        forAll(beta, cellI)
        {
            if (aspectRatio(cellI) >= aspectThresh_) beta[cellI] = 0.0;
        }
    }
    // Chevron helper → set beta=0 for chevron cells
    if (useChevron_)
    {
        forAll(beta, cellI)
        {
            if (beta[cellI] > 0.0)
            {
                if (betaChevron(cellI) < 0.5) beta[cellI] = 0.0;
            }
        }
    }
    // Flat curvature helper → reduce beta for strongly curved thin cells
    if (useFlat_)
    {
        forAll(beta, cellI)
        {
            beta[cellI] = min(beta[cellI], betaFlatCurvature(cellI));
        }
    }
    // LSQ ratio helper → reduce beta for poorly conditioned stencils
    if (useLsqRatio_)
    {
        forAll(beta, cellI)
        {
            beta[cellI] = min(beta[cellI], betaLsqQuality(cellI));
        }
    }
}

// ---- Limiter cores ----

template<class Type>
void hybridLsqGG<Type>::limitScalarGradient(const volScalarField& vsf, volVectorField& g) const
{
    if (limiter_ == limNone) return;

    const fvMesh& mesh = this->mesh();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const vectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    scalarField maxV(vsf.primitiveField());
    scalarField minV(vsf.primitiveField());

    forAll(owner, facei)
    {
        const label o = owner[facei];
        const label n = neighbour[facei];
        const scalar vo = vsf[o];
        const scalar vn = vsf[n];
        maxV[o] = max(maxV[o], vn); minV[o] = min(minV[o], vn);
        maxV[n] = max(maxV[n], vo); minV[n] = min(minV[n], vo);
    }

    // α per cell
    scalarField alpha(vsf.primitiveField().size(), 1.0);

    forAll(owner, facei)
    {
        const label o = owner[facei];
        const label n = neighbour[facei];

        const scalar dfo = (Cf[facei] - C[o]) & g[o];
        const scalar dfMaxo = maxV[o] - vsf[o];
        const scalar dfMino = minV[o] - vsf[o];
        scalar ro = (dfo > 0) ? ( (dfMaxo > SMALL) ? dfo/dfMaxo : 1 )
                              : ( (dfMino < -SMALL) ? dfo/dfMino : 1 );
        scalar aFo = 1.0;
        if (limiter_ == limVenkat)
        {
            aFo = (2*ro + 1)/(ro*(2*ro + 1) + 1);
        }
        else if (limiter_ == limVenkatMod)
        {
            const scalar denom = ro*(2*ro + 1) + 1;
            aFo = denom/(ro*denom + 1);
        }
        alpha[o] = min(alpha[o], max(aFo, scalar(0)));

        const scalar dfn = (Cf[facei] - C[n]) & g[n];
        const scalar dfMaxn = maxV[n] - vsf[n];
        const scalar dfMinn = minV[n] - vsf[n];
        scalar rn = (dfn > 0) ? ( (dfMaxn > SMALL) ? dfn/dfMaxn : 1 )
                              : ( (dfMinn < -SMALL) ? dfn/dfMinn : 1 );
        scalar aFn = 1.0;
        if (limiter_ == limVenkat)
        {
            aFn = (2*rn + 1)/(rn*(2*rn + 1) + 1);
        }
        else if (limiter_ == limVenkatMod)
        {
            const scalar denom = rn*(2*rn + 1) + 1;
            aFn = denom/(rn*denom + 1);
        }
        alpha[n] = min(alpha[n], max(aFn, scalar(0)));
    }

    g.primitiveFieldRef() *= alpha;
    g.correctBoundaryConditions();
}


template<class Type>
void hybridLsqGG<Type>::limitVectorGradient(const volVectorField& vsf, volTensorField& g) const
{
    if (limiter_ == limNone) return;

    const fvMesh& mesh = this->mesh();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    const vectorField& C = mesh.C();
    const surfaceVectorField& Cf = mesh.Cf();

    for (direction cmpt = 0; cmpt < vector::nComponents; ++cmpt)
    {
        // Build per-cell alpha (start at 1)
        scalarField alpha(mesh.nCells(), 1.0);

        // Precompute max/min per cell from component values
        scalarField maxV(vsf.primitiveField().size());
        scalarField minV(vsf.primitiveField().size());
        forAll(maxV, i) { const vector& v=vsf[i]; const scalar s=component(v,cmpt); maxV[i]=s; minV[i]=s; }

        forAll(owner, facei)
        {
            const label o = owner[facei];
            const label n = neighbour[facei];
            const scalar so = component(vsf[o], cmpt);
            const scalar sn = component(vsf[n], cmpt);
            if (sn > maxV[o]) { maxV[o] = sn; }
            if (sn < minV[o]) { minV[o] = sn; }
            if (so > maxV[n]) { maxV[n] = so; }
            if (so < minV[n]) { minV[n] = so; }
        }

        // Compute alpha per face (Venkat forms), reduce to cell min
        forAll(owner, facei)
        {
            const label o = owner[facei];
            const label n = neighbour[facei];

            // owner column vector (gradient of component cmpt)
            vector go(g[o][cmpt], g[o][cmpt+3], g[o][cmpt+6]);
            scalar dfo = (Cf[facei] - C[o]) & go;
            scalar dfMaxo = maxV[o] - component(vsf[o], cmpt);
            scalar dfMino = minV[o] - component(vsf[o], cmpt);
            scalar ro = (dfo > 0) ? ( (dfMaxo > SMALL) ? dfo/dfMaxo : 1 )
                                  : ( (dfMino < -SMALL) ? dfo/dfMino : 1 );
            scalar aFo = 1.0;
            if (limiter_ == limVenkat)
                aFo = (2*ro + 1)/(ro*(2*ro + 1) + 1);
            else if (limiter_ == limVenkatMod)
            {
                const scalar denom = ro*(2*ro + 1) + 1;
                aFo = denom/(ro*denom + 1);
            }
            if (aFo < alpha[o]) alpha[o] = max(aFo, scalar(0));

            // neighbour
            vector gn(g[n][cmpt], g[n][cmpt+3], g[n][cmpt+6]);
            scalar dfn = (Cf[facei] - C[n]) & gn;
            scalar dfMaxn = maxV[n] - component(vsf[n], cmpt);
            scalar dfMinn = minV[n] - component(vsf[n], cmpt);
            scalar rn = (dfn > 0) ? ( (dfMaxn > SMALL) ? dfn/dfMaxn : 1 )
                                  : ( (dfMinn < -SMALL) ? dfn/dfMinn : 1 );
            scalar aFn = 1.0;
            if (limiter_ == limVenkat)
                aFn = (2*rn + 1)/(rn*(2*rn + 1) + 1);
            else if (limiter_ == limVenkatMod)
            {
                const scalar denom = rn*(2*rn + 1) + 1;
                aFn = denom/(rn*denom + 1);
            }
            if (aFn < alpha[n]) alpha[n] = max(aFn, scalar(0));
        }

        // Apply alpha to the tensor column
        forAll(alpha, i)
        {
            g[i][cmpt]   *= alpha[i];
            g[i][cmpt+3] *= alpha[i];
            g[i][cmpt+6] *= alpha[i];
        }
    }
}

// ---- main calcGrad ----

template<class Type>
tmp< GeometricField< typename outerProduct<vector, Type>::type, fvPatchField, volMesh> >
hybridLsqGG<Type>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{
    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    const fvMesh& mesh = this->mesh();

    // Default LSQ: compute once; only compute Gauss if helpers requested mark beta==0 cells
    tmp<GradFieldType> tLS = fv::leastSquaresGrad<Type>(mesh).calcGrad(vsf, name + ":LS");
    GradFieldType& gLS = tLS.ref();

    // Compute β (0/1) using only requested helpers; if none active => all 1 (LSQ)
    scalarField beta(mesh.nCells(), 1.0);
    computeBeta(beta);

    // Fast path: no Gauss cells → return LSQ directly (with optional limiter below)
    if (findMin(beta) >= (1.0 - SMALL))
    {
        // Boundaries: Gauss-only to be consistent if desired
        tmp<GradFieldType> tGtmp = fv::gaussGrad<Type>(mesh).calcGrad(vsf, name + ":GGb");
        auto& gbf = gLS.boundaryFieldRef();
        const auto& ggbf = tGtmp().boundaryField();
        forAll(gbf, patchi) gbf[patchi] = ggbf[patchi];

        // Limiting (optional)
        if (enableLimiter_ && limiter_ != limNone)
        {
            if (pTraits<Type>::rank == 0)
                limitScalarGradient(reinterpret_cast<const volScalarField&>(vsf),
                                    reinterpret_cast<volVectorField&>(gLS));
            else if (pTraits<Type>::rank == 1)
                limitVectorGradient(reinterpret_cast<const volVectorField&>(vsf),
                                    reinterpret_cast<volTensorField&>(gLS));
        }
        fv::gaussGrad<Type>::correctBoundaryConditions(vsf, gLS);
        return tLS;
    }

    // Some cells require Gauss: compute Gauss once and overlay only those cells
    tmp<GradFieldType> tGG = fv::gaussGrad<Type>(mesh).calcGrad(vsf, name + ":GG");
    const GradFieldType& gG = tGG();

    // Overlay beta==0 cells with Gauss; keep LSQ for others
    forAll(beta, cellI)
    {
        if (beta[cellI] <= SMALL)
        {
            gLS[cellI] = gG[cellI];
        }
    }
    // Boundaries: Gauss
    {
        auto& gbf = gLS.boundaryFieldRef();
        const auto& ggbf = gG.boundaryField();
        forAll(gbf, patchi) gbf[patchi] = ggbf[patchi];
    }

    // Limiting (optional)
    if (enableLimiter_ && limiter_ != limNone)
    {
        if (pTraits<Type>::rank == 0)
            limitScalarGradient(reinterpret_cast<const volScalarField&>(vsf),
                                reinterpret_cast<volVectorField&>(gLS));
        else if (pTraits<Type>::rank == 1)
            limitVectorGradient(reinterpret_cast<const volVectorField&>(vsf),
                                reinterpret_cast<volTensorField&>(gLS));
    }

    fv::gaussGrad<Type>::correctBoundaryConditions(vsf, gLS);
    return tLS;
}

// Register

} // End namespace fv
} // End namespace Foam
