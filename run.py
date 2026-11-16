import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mol_dim': 2048,
    'label_emb_dim': 256,
    'feature_dim': 512,
    'gcn_hidden_dim': 512,
    'gcn_layers': 2,
    'dropout': 0.5,
    'leaky_slope': 0.2,
}

ODOR_LABELS = [
    "Alcoholic", "Almond", "Animalic", "Berry", "Buttery", "Caramel", "Cereal",
    "Cheesy", "Chemical", "Chocolate", "Citrus", "Coconut", "Coffee", "Earthy",
    "Fatty", "Fermented", "Floral", "Fruity", "Fuel", "Green", "Herbal", "Honey",
    "Mushroom", "Nutty", "Onion/Garlic", "Piney", "Rancid", "Resinous", "Roasted",
    "Rose", "Rubber", "Savory", "Smoky", "Solvent", "Spicy", "Sulfurous", "Sweet",
    "Tropical Fruit", "Vegetable", "Woody"
]


def get_fingerprint(smile, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.zeros((n_bits,), dtype=np.float32)
        mfgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        return mfgen.GetFingerprintAsNumPy(mol).astype(np.float32)
    except Exception:
        return np.zeros((n_bits,), dtype=np.float32)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output


class MLGCN(nn.Module):
    def __init__(self, num_classes, adj_matrix):
        super(MLGCN, self).__init__()
        self.adj = nn.Parameter(adj_matrix, requires_grad=False)

        self.mol_encoder = nn.Sequential(
            nn.Linear(CONFIG['mol_dim'], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(1024, CONFIG['feature_dim']),
            nn.Tanh()
        )

        self.label_emb = nn.Parameter(torch.Tensor(num_classes, CONFIG['label_emb_dim']))
        nn.init.xavier_uniform_(self.label_emb)

        self.gcn_layers = nn.ModuleList()
        self.leaky_relu = nn.LeakyReLU(CONFIG['leaky_slope'])
        self.gcn_layers.append(GraphConvolution(CONFIG['label_emb_dim'], CONFIG['gcn_hidden_dim']))
        for _ in range(CONFIG['gcn_layers'] - 2):
            self.gcn_layers.append(GraphConvolution(CONFIG['gcn_hidden_dim'], CONFIG['gcn_hidden_dim']))
        last_in_dim = CONFIG['gcn_hidden_dim'] if CONFIG['gcn_layers'] > 1 else CONFIG['label_emb_dim']
        self.gcn_layers.append(GraphConvolution(last_in_dim, CONFIG['feature_dim']))

    def forward(self, x):
        feature = self.mol_encoder(x)
        w = self.label_emb
        for i, layer in enumerate(self.gcn_layers):
            w = layer(w, self.adj)
            if i < len(self.gcn_layers) - 1:
                w = self.leaky_relu(w)
        return torch.mm(feature, w.t())


def load_model(model_path, num_classes, device):
    model = MLGCN(num_classes, torch.eye(num_classes)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_smiles(smiles_list, model, vocab, device, threshold=0.5):
    features = [get_fingerprint(s, CONFIG['mol_dim']) for s in smiles_list]
    features_tensor = torch.FloatTensor(np.array(features)).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(features_tensor)).cpu().numpy()

    results = []
    for i, prob in enumerate(probs):
        preds = {vocab[j]: float(p) for j, p in enumerate(prob) if p >= threshold}
        preds = dict(sorted(preds.items(), key=lambda item: item[1], reverse=True))
        results.append({'SMILES': smiles_list[i], 'Predictions': preds})
    return results


def main():
    parser = argparse.ArgumentParser(description="BF-MLGCN odor descriptor prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', type=str, help='SMILES string for single-molecule prediction')
    group.add_argument('--csv', type=str, help='Path to input CSV file for batch prediction')

    parser.add_argument('--model', type=str, default='best_model.pth', help='Path to trained model weights')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for odor labels')
    parser.add_argument('--output', type=str, default='output.csv', help='Output CSV file path')
    args = parser.parse_args()

    device = CONFIG['device']
    model = load_model(args.model, len(ODOR_LABELS), device)

    if args.smiles:
        res = predict_smiles([args.smiles], model, ODOR_LABELS, device, args.threshold)[0]
        preds_str = ", ".join([f"{k}({v:.4f})" for k, v in res['Predictions'].items()])
        print(f"SMILES: {args.smiles}\nPredicted Odors: {preds_str if preds_str else 'None (below threshold)'}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'

        smiles_list = df[smiles_col].dropna().tolist()
        results = predict_smiles(smiles_list, model, ODOR_LABELS, device, args.threshold)

        output_df = pd.DataFrame({
            'SMILES': [r['SMILES'] for r in results],
            'Predicted_Odors': [", ".join([f"{k}({v:.4f})" for k, v in r['Predictions'].items()]) for r in results]
        })

        if 'No' in df.columns:
            output_df.insert(0, 'No', df['No'])
        if 'CAS' in df.columns:
            output_df.insert(1, 'CAS', df['CAS'])

        output_df.to_csv(args.output, index=False)
        print(f"Batch prediction completed. Results saved to: {args.output}")


if __name__ == "__main__":
    main()
