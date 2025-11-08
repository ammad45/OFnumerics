/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
\*---------------------------------------------------------------------------*/

#include "blendedGradScheme.H"
#include "gradScheme.H"
#include "IStringStream.H"
#include "volFields.H"
#include "dictionary.H"
#include "string.H"
#include "regIOobject.H"
#include "surfaceFields.H"
#include "HashTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace fv
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
blendedGradScheme<Type>::blendedGradScheme
(
    const fvMesh& mesh,
    Istream& schemeData
)
:
    gradScheme<Type>(mesh),
    blendingFieldName_("gradSchemeBlending")
{
    // Defaults: instantiate leastSquares and Gauss linear
    {
        backgroundSpec_ = "leastSquares";
        blendedSpec_ = "Gauss linear";
        IStringStream is1(backgroundSpec_);
        backgroundScheme_ = gradScheme<Type>::New(mesh, is1);
        IStringStream is2(blendedSpec_);
        blendedScheme_ = gradScheme<Type>::New(mesh, is2);
    }

    // Prefer dictionary-style configuration for arbitrary-length schemes
    if (!schemeData.eof())
    {
        token t0(schemeData);
        if (t0.isPunctuation() && t0.pToken() == token::BEGIN_BLOCK)
        {
            schemeData.putBack(t0);
            dictionary dict(schemeData);

            if (const entry* e1 = dict.findEntry("scheme1", keyType::LITERAL))
            {
                ITstream& es = e1->stream();
                backgroundSpec_.clear();
                while (!es.eof())
                {
                    token tk(es);
                    if (tk.isWord()) backgroundSpec_ += (backgroundSpec_.empty()? "" : " ") + tk.wordToken();
                    else if (tk.isNumber()) backgroundSpec_ += (backgroundSpec_.empty()? "" : " ") + name(tk.number());
                    else if (tk.isString()) backgroundSpec_ += (backgroundSpec_.empty()? "" : " ") + tk.stringToken();
                }
                IStringStream is(backgroundSpec_);
                backgroundScheme_.reset(gradScheme<Type>::New(mesh, is));
            }

            if (const entry* e2 = dict.findEntry("scheme2", keyType::LITERAL))
            {
                ITstream& es = e2->stream();
                blendedSpec_.clear();
                while (!es.eof())
                {
                    token tk(es);
                    if (tk.isWord()) blendedSpec_ += (blendedSpec_.empty()? "" : " ") + tk.wordToken();
                    else if (tk.isNumber()) blendedSpec_ += (blendedSpec_.empty()? "" : " ") + name(tk.number());
                    else if (tk.isString()) blendedSpec_ += (blendedSpec_.empty()? "" : " ") + tk.stringToken();
                }
                IStringStream is(blendedSpec_);
                blendedScheme_.reset(gradScheme<Type>::New(mesh, is));
            }
        }
        else
        {
            // Fallback: positional tokens
            schemeData.putBack(t0);
            // First token -> scheme1, remainder -> scheme2 (best-effort fallback)
            backgroundSpec_.clear();
            blendedSpec_.clear();
            if (!schemeData.eof())
            {
                token t1(schemeData);
                if (!t1.isPunctuation())
                {
                    if (t1.isWord()) backgroundSpec_ = t1.wordToken();
                    else if (t1.isNumber()) backgroundSpec_ = name(t1.number());
                    else if (t1.isString()) backgroundSpec_ = t1.stringToken();
                }
            }
            while (!schemeData.eof())
            {
                token tk(schemeData);
                if (tk.isPunctuation()) break;
                if (tk.isWord()) blendedSpec_ += (blendedSpec_.empty()? "" : " ") + tk.wordToken();
                else if (tk.isNumber()) blendedSpec_ += (blendedSpec_.empty()? "" : " ") + name(tk.number());
                else if (tk.isString()) blendedSpec_ += (blendedSpec_.empty()? "" : " ") + tk.stringToken();
            }
            if (backgroundSpec_.empty()) backgroundSpec_ = "leastSquares";
            if (blendedSpec_.empty()) blendedSpec_ = "Gauss linear";
            IStringStream is1(backgroundSpec_);
            backgroundScheme_.reset(gradScheme<Type>::New(mesh, is1));
            IStringStream is2(blendedSpec_);
            blendedScheme_.reset(gradScheme<Type>::New(mesh, is2));
        }
    }

    Info<< "blendedGradScheme: background=" << backgroundSpec_
        << ", blended=" << blendedSpec_
        << ", field=" << blendingFieldName_ << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class Type>
tmp< GeometricField< typename outerProduct<vector, Type>::type, fvPatchField, volMesh> >
blendedGradScheme<Type>::calcGrad
(
    const GeometricField<Type, fvPatchField, volMesh>& vsf,
    const word& name
) const
{
    typedef typename outerProduct<vector, Type>::type GradType;
    typedef GeometricField<GradType, fvPatchField, volMesh> GradFieldType;

    const fvMesh& mesh = this->mesh();

    // Look up blending field (and try to read/register if missing)
    const volScalarField* blendingPtr = mesh.findObject<volScalarField>(blendingFieldName_);
    if (!blendingPtr)
    {
        // Try to locate in the most suitable time directory
        const word inst = mesh.time().findInstance(mesh.dbDir(), blendingFieldName_);
        IOobject fieldHeader
        (
            blendingFieldName_,
            inst,
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE,
            IOobject::REGISTER
        );

        if (fieldHeader.typeHeaderOk<volScalarField>(true, true, false))
        {
            auto* vfPtr = new volScalarField(fieldHeader, mesh);
            regIOobject::store(vfPtr);
            blendingPtr = vfPtr;
        }
    }

    if (!blendingPtr)
    {
        WarningInFunction
            << "Blending field '" << blendingFieldName_
            << "' not found. Using background scheme only." << endl;

        return backgroundScheme_().calcGrad(vsf, name);
    }

    const volScalarField& blending = *blendingPtr;
    const scalarField& beta = blending.primitiveField();

    // Compute both gradients from stored schemes
    tmp<GradFieldType> tbackground = backgroundScheme_().calcGrad(vsf, name + ":background");
    tmp<GradFieldType> tblended = blendedScheme_().calcGrad(vsf, name + ":blended");

    const GradFieldType& gBackground = tbackground();
    const GradFieldType& gBlended = tblended();

    // Create result and blend smoothly
    tmp<GradFieldType> tresult
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
            dimensioned<GradType>(vsf.dimensions()/dimLength, Zero)
        )
    );
    GradFieldType& g = tresult.ref();

    // Smooth blending: g = (1-β)*background + β*blended
    forAll(g, cellI)
    {
        const scalar b = max(min(beta[cellI], scalar(1)), scalar(0));  // Clamp to [0,1]
        g[cellI] = (1.0 - b)*gBackground[cellI] + b*gBlended[cellI];
    }

    // Boundaries: use blended scheme (or could blend based on boundary field if needed)
    g.boundaryFieldRef() = gBlended.boundaryField();

    // Correct boundary conditions
    g.correctBoundaryConditions();

    return tresult;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace fv
} // End namespace Foam

// ************************************************************************* //

