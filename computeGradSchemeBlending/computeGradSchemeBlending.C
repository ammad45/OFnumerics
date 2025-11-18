/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    computeGradSchemeBlending

Group
    grpPreProcessingUtilities

Description
    Precomputes gradient scheme blending factors once per simulation.
    Reads gradSchemeBlendingDict and writes a volScalarField named
    gradSchemeBlending (0..1) that indicates the blending weight:
    - 0: use background scheme (default: leastSquares)
    - 1: use blended scheme (default: gauss)

    The blending factors are computed using the same quality criteria
    as hybridLsqGG: aspect ratio, chevron detection, flat curvature,
    and LSQ tensor quality.

Usage
    Run before simulation:
        computeGradSchemeBlending

    Dictionary: system/gradSchemeBlendingDict
        {
            // Optional helpers (same as hybridLsqGG)
            aspect         10.0;      // aspect ratio threshold
            chevron        on;         // enable chevron detection
            flat           0.2 2.0;    // NCF decay
            lsq            2.0;        // eigenvalue ratio threshold
        }

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "mathematicalConstants.H"
#include "cellQuality.H"

// Helper functions (reused from hybridLsqGG logic)

scalar aspectRatio(const fvMesh& mesh, const label cellI)
{
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

scalar betaChevron(const fvMesh& mesh, const label cellI)
{
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
            ds = Cf[fI] - C0;
        }
        const vector dx = Cf[fI] - C0;
        const vector Af = Sf[fI];
        const scalar Af_dx = Af & dx;
        const scalar Af_ds = Af & ds;
        vector dp = ((Af_ds != 0) ? (Af_dx/Af_ds)*ds : vector::zero) - dx;

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

scalar maxSkewAngleDeg(const fvMesh& mesh, const label cellI)
{
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
        const scalar ang = Foam::acos(max(min(c, scalar(1)), scalar(-1)))
                         * (180.0/Foam::constant::mathematical::pi);
        if (ang > amax) amax = ang;
    }
    return amax;
}

// maxNonOrthAngleDeg removed; use cellQuality.nonOrthogonality() instead

scalar betaFlatCurvature(const fvMesh& mesh, const label cellI, const scalar flatNCF, const scalar flatDecay)
{
    const scalar ang = maxSkewAngleDeg(mesh, cellI);
    const scalar t = Foam::tan(ang*Foam::constant::mathematical::pi/180.0);
    const scalar ar = aspectRatio(mesh, cellI);
    const scalar tau = flatNCF*ar;
    if (t <= tau) return 1.0;
    const scalar t2 = flatDecay*tau;
    if (t >= t2) return 0.0;
    return max(scalar(0), 1.0 - (t - tau)/max(t2 - tau, SMALL));
}

scalar betaLsqQuality(const fvMesh& mesh, const label cellI, const scalar lsqEigenRatioMin)
{
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
    return (ratio >= lsqEigenRatioMin) ? scalar(0) : scalar(1);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "addCheckCaseOptions.H"
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    Info<< "Reading gradSchemeBlendingDict\n" << endl;

    IOdictionary blendingDict
    (
        IOobject
        (
            "gradSchemeBlendingDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    );

    // Parse optional helpers
    bool useAspect = blendingDict.found("aspect");
    scalar aspectThresh = useAspect ? blendingDict.get<scalar>("aspect") : 10.0;

    bool useChevron = blendingDict.found("chevron");
    word chevronVal = useChevron ? blendingDict.get<word>("chevron") : word("off");
    useChevron = (chevronVal == "on" || chevronVal == "true" || chevronVal == "1");

    bool useFlat = blendingDict.found("flat");
    scalar flatNCF = 0.2;
    scalar flatDecay = 2.0;
    if (useFlat)
    {
        // Find the 'flat' entry and interpret as dict, stream (list), or scalar
        const entry* ePtr = blendingDict.findEntry("flat", keyType::LITERAL);
        if (ePtr)
        {
            if (ePtr->isDict())
            {
                const dictionary& flatDict = ePtr->dict();
                flatNCF = flatDict.get<scalar>("NCF");
                flatDecay = flatDict.getOrDefault<scalar>("decay", 2.0);
            }
            else if (ePtr->isStream())
            {
                Istream& is = ePtr->stream();
                is >> flatNCF;
                if (!is.eof())
                {
                    token t(is);
                    if (t.isNumber()) flatDecay = scalar(t.number());
                }
            }
            else
            {
                flatNCF = blendingDict.get<scalar>("flat");
            }
        }
    }

    bool useLsqRatio = blendingDict.found("lsq");
    scalar lsqEigenRatioMin = useLsqRatio ? blendingDict.get<scalar>("lsq") : 2.0;

    // Optional smoothing of the resulting blending field
    bool doSmooth = blendingDict.found("smooth");
    scalar smoothW = 1.0;     // 1.0 -> pure neighbour average, 0.0 -> no change
    label smoothPasses = 1;   // number of smoothing passes
    if (doSmooth)
    {
        const entry* ePtr = blendingDict.findEntry("smooth", keyType::LITERAL);
        if (ePtr)
        {
            if (ePtr->isDict())
            {
                const dictionary& sDict = ePtr->dict();
                smoothW = sDict.getOrDefault<scalar>("weight", 1.0);
                smoothPasses = sDict.getOrDefault<label>("passes", 1);
            }
            else if (ePtr->isStream())
            {
                // If a single scalar is provided, treat as weight
                Istream& is = ePtr->stream();
                token tk(is);
                if (tk.isNumber()) { smoothW = scalar(tk.number()); }
            }
        }
        smoothW = max(min(smoothW, scalar(1)), scalar(0));
        smoothPasses = max(smoothPasses, 1);
    }

    // Optional gating: force background (beta=0) on poor-quality cells
    bool useNonOrthGate = blendingDict.found("nonOrth");
    scalar nonOrthTolDeg = useNonOrthGate ? blendingDict.get<scalar>("nonOrth") : 0.0; // degrees
    bool useSkewGate = blendingDict.found("skew");
    scalar skewTol = useSkewGate ? blendingDict.get<scalar>("skew") : 0.0; // meshQuality skewness (unitless)

    Info<< "Blending criteria:" << nl
        << "  aspect: " << (useAspect ? "on (threshold: " + name(aspectThresh) + ")" : "off") << nl
        << "  chevron: " << (useChevron ? "on" : "off") << nl
        << "  flat: " << (useFlat ? "on (NCF: " + name(flatNCF) + ", decay: " + name(flatDecay) + ")" : "off") << nl
        << "  lsq: " << (useLsqRatio ? "on (ratio: " + name(lsqEigenRatioMin) + ")" : "off") << nl
        << "  nonOrth gate: " << (useNonOrthGate ? "on (deg: " + name(nonOrthTolDeg) + ")" : "off") << nl
        << "  skew gate: " << (useSkewGate ? "on (skewness: " + name(skewTol) + ")" : "off") << nl
        << endl;

    // Create blending field (0..1): 0 = background scheme, 1 = blended scheme
    volScalarField blending
    (
        IOobject
        (
            "gradSchemeBlending",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar(dimless, 0.0),  // Default: all background scheme
        fvPatchFieldBase::zeroGradientType()
    );

    scalarField& beta = blending.primitiveFieldRef();
    beta = 0.0;  // Start with all background scheme (LSQ)

    Info<< "Computing blending factors..." << endl;

    // Apply helpers (same logic as hybridLsqGG)
    // Note: In hybridLsqGG, beta=1 means LSQ, beta=0 means Gauss
    // Here, beta=0 means background (LSQ), beta=1 means blended (Gauss)
    // So we invert the logic: where hybridLsqGG sets beta=0, we set beta=1
    if (useAspect)
    {
        Info<< "  Applying aspect ratio criterion..." << endl;
        forAll(beta, cellI)
        {
            if (aspectRatio(mesh, cellI) >= aspectThresh) beta[cellI] = 1.0;
        }
    }

    if (useChevron)
    {
        Info<< "  Applying chevron detection..." << endl;
        forAll(beta, cellI)
        {
            if (betaChevron(mesh, cellI) < 0.5) beta[cellI] = 1.0;
        }
    }

    if (useFlat)
    {
        Info<< "  Applying flat curvature criterion..." << endl;
        forAll(beta, cellI)
        {
            const scalar bFlat = betaFlatCurvature(mesh, cellI, flatNCF, flatDecay);
            // Invert: where bFlat=0 (use Gauss), set beta=1
            if (bFlat < 0.5) beta[cellI] = 1.0;
        }
    }

    if (useLsqRatio)
    {
        Info<< "  Applying LSQ quality criterion..." << endl;
        forAll(beta, cellI)
        {
            const scalar bLsq = betaLsqQuality(mesh, cellI, lsqEigenRatioMin);
            // Invert: where bLsq=0 (use Gauss), set beta=1
            if (bLsq < 0.5) beta[cellI] = 1.0;
        }
    }

    // Optional neighbour-averaging smooth
    if (doSmooth)
    {
        Info<< "  Smoothing blending field (weight=" << smoothW
            << ", passes=" << smoothPasses << ")..." << endl;

        const labelUList& owner = mesh.owner();
        const labelUList& neighbour = mesh.neighbour();

        for (label pass = 0; pass < smoothPasses; ++pass)
        {
            scalarField sum(mesh.nCells(), 0.0);
            scalarField cnt(mesh.nCells(), 0.0);

            forAll(owner, facei)
            {
                const label o = owner[facei];
                const label n = neighbour[facei];
                sum[o] += beta[n]; cnt[o] += 1.0;
                sum[n] += beta[o]; cnt[n] += 1.0;
            }

            forAll(beta, cellI)
            {
                if (cnt[cellI] > VSMALL)
                {
                    const scalar avg = sum[cellI]/cnt[cellI];
                    beta[cellI] = (1.0 - smoothW)*beta[cellI] + smoothW*avg;
                }
            }
        }
    }

    // Enforce gating after all operations (strong override to background)
    if (useNonOrthGate || useSkewGate)
    {
        Info<< "  Applying mesh-quality gates (forcing background where poor)..." << endl;
        // Compute per-cell mesh quality fields once
        cellQuality meshQuality(mesh);
        scalarField nonOrth, skew;
        if (useNonOrthGate) nonOrth = meshQuality.nonOrthogonality(); // degrees per cell
        if (useSkewGate)    skew    = meshQuality.skewness();         // unitless per cell
        label forced = 0;
        forAll(beta, cellI)
        {
            bool poor = false;
            if (useNonOrthGate && nonOrth[cellI] > nonOrthTolDeg) poor = true;
            if (!poor && useSkewGate && skew[cellI] > skewTol)    poor = true;
            if (poor)
            {
                if (beta[cellI] != 0.0) { ++forced; }
                beta[cellI] = 0.0;
            }
        }
        Info<< "    Forced background in " << forced << " cells." << endl;
    }

    // Statistics
    label nBackground = 0;
    scalar minB = GREAT;
    scalar maxB = -GREAT;
    forAll(beta, i)
    {
        const scalar b = beta[i];
        if (b < 0.5) ++nBackground;
        if (b < minB) minB = b;
        if (b > maxB) maxB = b;
    }
    const label nBlended = mesh.nCells() - nBackground;
    Info<< nl
        << "Blending statistics:" << nl
        << "  Background scheme cells: " << nBackground << nl
        << "  Blended scheme cells: " << nBlended << nl
        << "  Min blending: " << minB << nl
        << "  Max blending: " << maxB << nl
        << endl;

    blending.write();

    Info<< "Blending field written to: " << blending.objectPath() << nl << endl;

    return 0;
}

// ************************************************************************* //

