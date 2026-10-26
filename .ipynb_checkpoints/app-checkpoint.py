from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import base64
from io import BytesIO

app = Flask(__name__)

# -------------------------------
# Disable caching so HTML updates instantly
# -------------------------------
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(r"C:\Users\ASUS\hiv_dataset\HIV.csv")
smiles_list = df["smiles"].tolist()
label_dict = dict(zip(df["smiles"], df["HIV_active"]))

# -------------------------------
# Load trained model
# -------------------------------
model = pickle.load(open(r"C:\Users\ASUS\hiv_dataset\model.pkl", "rb"))

# -------------------------------
# Morgan Fingerprint Featurizer
# -------------------------------
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.reshape(1, -1)

# -------------------------------
# Convert molecule to PNG Base64
# -------------------------------
def mol_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(250, 250))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -------------------------------
# Functional Groups (SMARTS)
# -------------------------------
functional_groups = {
    "Nitro Group": "[N+](=O)[O-]",
    "Amide": "C(=O)N",
    "Aromatic Ring": "a1aaaaa1",
    "Halogen": "[F,Cl,Br,I]",
    "Hydroxyl": "[OX2H]"
}

# -------------------------------
# Home (Dropdown Molecule Prediction)
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    selected = None
    prediction = None
    actual = None
    image_data = None

    if request.method == "POST":
        selected = request.form.get("selected_smiles")

        X = featurize(selected)
        if X is not None:
            prediction = int(model.predict(X)[0])
        actual = label_dict.get(selected, "Unknown")
        image_data = mol_to_image(selected)

    return render_template(
        "index.html",
        smiles_list=smiles_list,
        selected_smiles=selected,
        prediction=prediction,
        actual=actual,
        image=image_data,
        functional_groups=functional_groups
    )

# -------------------------------
# Search Functional Groups
# -------------------------------
@app.route("/search_group", methods=["POST"])
def search_group():
    group_key = request.form.get("group")
    smarts = functional_groups[group_key]
    patt = Chem.MolFromSmarts(smarts)

    matches = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol and mol.HasSubstructMatch(patt):
            matches.append(s)

    return render_template(
        "group_results.html",
        group=group_key,
        matches=matches
    )

# -------------------------------
# Manual SMILES Search
# -------------------------------
@app.route("/manual", methods=["GET", "POST"])
def manual():
    smi = None
    prediction = None
    actual = None
    image_data = None
    error = None

    if request.method == "POST":
        smi = request.form.get("manual_smiles")
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            error = "Invalid SMILES entered!"
        else:
            X = featurize(smi)
            prediction = int(model.predict(X)[0])
            actual = label_dict.get(smi, "Not in dataset")
            image_data = mol_to_image(smi)

    return render_template(
        "manual.html",
        smi=smi,
        prediction=prediction,
        actual=actual,
        image=image_data,
        error=error
    )

@app.route("/active")
def active_list():
    active_mols = df[df["HIV_active"] == 1]["smiles"].tolist()
    return render_template("active.html", active_mols=active_mols)


# Run App
if __name__ == "__main__":
    app.run(debug=True)
