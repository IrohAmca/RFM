import torch

from rfm.analysis.clustering import FeatureClustering
from rfm.sae.model import SparseAutoEncoder

def test_feature_clustering_loads_model(tmp_path):
    # Dummy SAE creation
    model = SparseAutoEncoder(16, 64)
    model.W_dec.data = torch.randn(64, 16)
    
    # Save dummy model
    model_path = tmp_path / "dummy_sae.pt"
    torch.save({"state_dict": model.state_dict()}, model_path)
    
    fc = FeatureClustering(model_path)
    assert fc.W_dec is not None
    assert fc.W_dec.shape == (64, 16)

def test_feature_clustering_similarity_matrix(tmp_path):
    model = SparseAutoEncoder(16, 64)
    model.W_dec.data = torch.randn(64, 16)
    model_path = tmp_path / "dummy_sae.pt"
    torch.save(model.state_dict(), model_path)
    
    fc = FeatureClustering(model_path)
    sim_mat = fc.compute_similarity_matrix(normalize=True)
    assert sim_mat.shape == (64, 64)
    # diagonal should be close to 1
    assert torch.allclose(torch.diag(sim_mat), torch.ones(64), atol=1e-4)

def test_feature_clustering_find_similar(tmp_path):
    model = SparseAutoEncoder(16, 64)
    # Make feature 0 and feature 1 identical
    model.W_dec.data = torch.randn(64, 16)
    model.W_dec.data[1] = model.W_dec.data[0]
    
    model_path = tmp_path / "dummy_sae.pt"
    torch.save(model.state_dict(), model_path)
    
    fc = FeatureClustering(model_path)
    similar = fc.get_most_similar_features(feature_id=0, top_k=5)
    
    # First top similar should be 1
    assert similar[0][0] == 1
    assert similar[0][1] > 0.99

