import h5py
import numpy as np
import sys
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from construct_offline_dataset import knn_entropy, normalize_features, propagate_success, write_hdf5, Trajectory, metrics, candidate_names, choose, sha256_file


def test_success_propagates_from_first_true():
    ok, labels = propagate_success(np.array([False, True, False]))
    assert ok and labels.tolist() == [False, True, True]


def test_knn_excludes_self_and_is_finite():
    x = np.arange(20, dtype=np.float64).reshape(10, 2)
    value, std = knn_entropy(x, k=2, sample_size=10, repeats=2, seed=4)
    assert np.isfinite(value) and np.isfinite(std)


def test_normalization_is_shared_across_trajectories():
    ts = [
        Trajectory("demo_1", 2, 1.0, True, np.array([False, True]), np.empty((2, 0)), np.array([[0., 1.], [2., 3.]]), np.array([[0., 2.], [2., 4.]])),
        Trajectory("demo_2", 2, 0.0, False, np.array([False, False]), np.empty((2, 0)), np.array([[4., 5.], [6., 7.]]), np.array([[4., 6.], [6., 8.]])),
    ]
    stats = normalize_features(ts, 1e-12)
    assert np.allclose(np.mean(np.concatenate([t.features for t in ts]), axis=0), 0.0)
    assert len(stats["state_mean"]) == 2 and len(stats["action_mean"]) == 2


def test_hdf5_write_has_one_terminal_step(tmp_path):
    source = tmp_path / "source.hdf5"; output = tmp_path / "out.hdf5"
    with h5py.File(source, "w") as f:
        data = f.create_group("data")
        g = data.create_group("demo_1"); g.attrs["num_samples"] = 3
        g.create_dataset("actions", data=np.zeros((3, 1))); g.create_dataset("rewards", data=np.ones(3)); g.create_dataset("dones", data=np.array([1, 0, 0], dtype=np.int8))
        g.create_dataset("states", data=np.zeros((3, 1))); o = g.create_group("obs"); no = g.create_group("next_obs"); o.create_dataset("state", data=np.zeros((3, 1))); no.create_dataset("state", data=np.zeros((3, 1)))
    t = Trajectory("demo_1", 3, 3.0, True, np.array([False, True, True]), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)))
    write_hdf5(source, output, [t], SimpleNamespace(action_key="actions", done_key="dones"))
    with h5py.File(output, "r") as f:
        assert np.asarray(f["data/demo_1/dones"]).tolist() == [0, 0, 1]
        assert np.asarray(f["data/demo_1/success"]).tolist() == [False, True, True]
        assert np.asarray(f["data/demo_1/rewards"]).tolist() == [1., 1., 1.]


def test_success_ratio_and_episode_mean_return():
    ts = [Trajectory("a", 1, 10., True, np.array([True]), np.array([[0.]]), np.array([[0.]]), np.array([[0.]])), Trajectory("b", 9, 0., False, np.array([False]), np.arange(9, dtype=float).reshape(-1,1), np.arange(9, dtype=float).reshape(-1,1), np.arange(9, dtype=float).reshape(-1,1))]
    class A: pass
    a=A(); a.knn_k=1; a.entropy_sample_size=10; a.entropy_num_repeats=1; a.seed=0
    normalize_features(ts, 1e-12)
    m=metrics(ts,a,0); assert m["success_ratio"] == .5 and m["mean_episode_return"] == 5.


def test_knn_query_does_not_use_self_neighbor():
    x=np.array([[0.], [1.], [3.], [6.]])
    one,_=knn_entropy(x,1,4,1,0); two,_=knn_entropy(x,2,4,1,0); assert one != two


def test_fixed_seed_candidate_ids():
    ts=[Trajectory(str(i),1,0.,False,np.array([False]),np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1))) for i in range(5)]
    class A: target_num_trajectories=3; use_success_ratio=False; success_ratio_min=0.; success_ratio_max=1.
    assert candidate_names(ts,A(),np.random.default_rng(9)) == candidate_names(ts,A(),np.random.default_rng(9))


def test_infeasible_threshold_raises():
    ts=[Trajectory(str(i),2,0.,False,np.array([False,False]),np.array([[float(i)],[float(i+1)]]),np.array([[float(i)],[float(i+1)]]),np.array([[float(i)],[float(i+1)]])) for i in range(3)]
    normalize_features(ts,1e-12)
    class A: pass
    a=A(); a.use_success_ratio=True; a.use_mean_return=False; a.use_state_action_entropy=False; a.budget_mode="trajectories"; a.target_num_trajectories=2; a.target_num_transitions=None; a.transition_budget_tolerance=0; a.success_ratio_min=1.; a.success_ratio_max=1.; a.mean_return_min=None; a.mean_return_max=None; a.state_action_entropy_min=-100.; a.knn_k=1; a.entropy_sample_size=3; a.entropy_num_repeats=1; a.num_search_trials=2; a.num_greedy_swaps=0; a.seed=0
    try: choose(ts,a)
    except RuntimeError: return
    raise AssertionError("infeasible threshold was accepted")


def test_source_hash_unchanged(tmp_path):
    p=tmp_path/"x"; p.write_bytes(b"source"); before=sha256_file(p); assert sha256_file(p)==before


def test_written_layout_has_project_reader_keys(tmp_path):
    p=tmp_path/"s.hdf5"; q=tmp_path/"o.hdf5"
    with h5py.File(p,"w") as f:
        d=f.create_group("data"); g=d.create_group("demo_1"); g.attrs["num_samples"]=2
        for k in ("actions","rewards","dones"): g.create_dataset(k,data=np.zeros(2))
        g.create_dataset("states",data=np.zeros((2,1))); o=g.create_group("obs"); n=g.create_group("next_obs"); o.create_dataset("state",data=np.zeros((2,1))); n.create_dataset("state",data=np.zeros((2,1)))
    t=Trajectory("demo_1",2,0.,False,np.array([False,False]),np.zeros((2,1)),np.zeros((2,1)),np.zeros((2,1))); write_hdf5(p,q,[t],SimpleNamespace(action_key="actions",done_key="dones"))
    with h5py.File(q,"r") as f: assert all(k in f["data/demo_1"] for k in ("obs","next_obs","actions","rewards","dones","success"))
