# Issue #32 調査レポート: FPFHとKDTreeがRANSACに与える影響

## 概要

Issue #32は、FPFH（Fast Point Feature Histograms）を用いたRANSACと用いないRANSACの差を調べることを目的とした機能強化要求です。

## Issue詳細

### タイトル
[✨Enhancement] FPFHとKDTreeがRANSACに与える影響

### 作成者
- @Stone5656

### アサイン
- @blazex60
- @Stone5656

### ラベル
- enhancement（機能強化の場合）

### 調査項目（Issue #32より）

1. [ ] FPFHの結果がRANSACに入れられるかどうか
2. [ ] FPFHにいれるKDTreeのparamsが結果にどういった影響を与えるか
3. [ ] FPFHを用いた場合のfitnessの差
4. [ ] FPFHを用いた場合と用いなかった場合の処理速度の差（FPFHの実行時間とRANSACの実行時間を別で取得したい）

---

## 現在のコード実装分析

### 1. 現在のRANSAC実装 (`src/matcher/ransac.py`)

現在の実装では、`registration_ransac_based_on_feature_matching` を使用しています：

```python
def global_registration(
    src: Ply,
    tgt: Ply,
    voxel_size: float,
    iteration: int = 30,
) -> pipelines.registration.RegistrationResult:
    dist_thresh = voxel_size * 1.5
    result = pipelines.registration.registration_ransac_based_on_feature_matching(
        src.pcd_down,
        tgt.pcd_down,
        src.pcd_fpfh,
        tgt.pcd_fpfh,
        True,
        dist_thresh,
        pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
    )
    logger.info("Global RANSAC result: %s", result)
    return result
```

**重要点**: 現在の実装は既にFPFH特徴量を使用しています（`src.pcd_fpfh`, `tgt.pcd_fpfh`）。

### 2. FPFH計算 (`src/ply/ply.py`)

FPFH特徴量は `Ply` クラスの `_preprocess` メソッドで計算されています：

```python
def _preprocess(
    self,
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30),
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, pcd_fpfh
```

**現在のKDTreeパラメータ**:
- 法線推定: `radius=voxel_size * 2`, `max_nn=30`
- FPFH計算: `radius=voxel_size * 5`, `max_nn=100`

---

## Open3D API リファレンス

### 1. FPFH (Fast Point Feature Histograms)

**API**: `open3d.pipelines.registration.compute_fpfh_feature`

URL: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.compute_fpfh_feature.html

**シグネチャ**:
```python
compute_fpfh_feature(input, search_param)
```

**パラメータ**:
- `input`: 入力点群（法線推定が必要）
- `search_param`: KDTree検索パラメータ

**戻り値**:
- `Feature`: 33次元のFPFH特徴量

### 2. KDTreeSearchParam

**API**: `open3d.geometry.KDTreeSearchParam`

URL: https://www.open3d.org/docs/release/python_api/open3d.geometry.KDTreeSearchParam.html

**利用可能なサブクラス**:
1. `KDTreeSearchParamHybrid(radius, max_nn)` - 半径と最大近傍点数のハイブリッド検索
2. `KDTreeSearchParamKNN(knn)` - K最近傍検索
3. `KDTreeSearchParamRadius(radius)` - 半径検索

### 3. RANSAC (Registration)

**API 1**: `registration_ransac_based_on_feature_matching` (特徴量ベース)

URL: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_ransac_based_on_feature_matching.html

**シグネチャ**:
```python
registration_ransac_based_on_feature_matching(
    source, target, source_feature, target_feature,
    mutual_filter, max_correspondence_distance,
    estimation_method, ransac_n, checkers, criteria
)
```

**API 2**: `registration_ransac_based_on_correspondence` (対応点ベース)

URL: https://www.open3d.org/docs/release/python_api/open3d.pipelines.registration.registration_ransac_based_on_correspondence.html

**注意**: Issue本文に記載されているRANSACのAPIリンクの一つは、FPFHのAPIリンクと同じになっているエラーがあります。

---

## 調査項目への回答と提案

### 1. FPFHの結果がRANSACに入れられるかどうか

**回答**: **はい、既に入れられています。**

現在の実装（`src/matcher/ransac.py`）では、`registration_ransac_based_on_feature_matching` 関数を使用しており、FPFH特徴量(`src.pcd_fpfh`, `tgt.pcd_fpfh`)を引数として渡しています。

### 2. FPFHにいれるKDTreeのparamsが結果にどういった影響を与えるか

**調査が必要な項目**:

KDTreeパラメータの影響を調べるためには、以下のパラメータを変化させて実験する必要があります：

| パラメータ | 現在の値 | 試すべき範囲 |
|-----------|----------|-------------|
| 法線推定 radius | voxel_size * 2 | voxel_size * 1〜5 |
| 法線推定 max_nn | 30 | 10〜100 |
| FPFH計算 radius | voxel_size * 5 | voxel_size * 2〜10 |
| FPFH計算 max_nn | 100 | 30〜200 |

**予想される影響**:
- `radius` が小さい: より局所的な特徴、ノイズに敏感
- `radius` が大きい: より広域的な特徴、細かい構造を見逃す可能性
- `max_nn` が小さい: 計算が速いが精度低下
- `max_nn` が大きい: 精度向上するが計算コスト増加

### 3. FPFHを用いた場合のfitnessの差

**比較実験の提案**:

1. **FPFHあり（現在の実装）**: `registration_ransac_based_on_feature_matching`
2. **FPFHなし**: `registration_ransac_based_on_correspondence` または単純なICP

**注意**: Open3Dでは「FPFHなしのRANSAC」は直接サポートされていません。特徴量なしでの対応付けには、以下の方法が考えられます：
- 点ベースの対応付け（距離ベース）
- 法線ベースの対応付け
- カスタム特徴量の使用

### 4. FPFHを用いた場合と用いなかった場合の処理速度の差

**計測ポイントの提案**:

```python
import time

# FPFH計算時間
start_fpfh = time.perf_counter()
pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, search_param)
fpfh_time = time.perf_counter() - start_fpfh

# RANSAC計算時間
start_ransac = time.perf_counter()
result = registration_ransac_based_on_feature_matching(...)
ransac_time = time.perf_counter() - start_ransac

# 合計時間
total_time = fpfh_time + ransac_time
```

---

## 実装提案

Issue #32を解決するために、以下の実装を提案します：

### 1. パラメータ化されたFPFH/KDTree設定

```python
# src/ply/ply.py の修正案
class Ply:
    def __init__(
        self, 
        path: Path, 
        voxel_size: float,
        normal_radius_factor: float = 2.0,  # NEW
        normal_max_nn: int = 30,  # NEW
        fpfh_radius_factor: float = 5.0,  # NEW
        fpfh_max_nn: int = 100,  # NEW
    ) -> None:
        ...
```

### 2. ベンチマーク用スクリプトの作成

```python
# src/benchmark/fpfh_benchmark.py
def benchmark_fpfh_params(src_path, tgt_path, param_combinations):
    """FPFHパラメータの影響を計測するベンチマーク"""
    results = []
    for params in param_combinations:
        # FPFH計算時間
        # RANSAC計算時間
        # fitness, inlier_rmse
        results.append({...})
    return results
```

### 3. 比較実験用関数

```python
# src/matcher/ransac.py への追加案
def global_registration_without_fpfh(src, tgt, voxel_size, iteration=30):
    """FPFHを使わないRANSAC（対応点ベース）"""
    # registration_ransac_based_on_correspondence を使用
    pass
```

---

## 関連ファイル一覧

| ファイル | 説明 |
|---------|------|
| `src/ply/ply.py` | FPFH計算、KDTreeパラメータ設定 |
| `src/matcher/ransac.py` | RANSAC実装 |
| `src/matcher/icp.py` | ICP実装 |
| `src/main.py` | メインエントリーポイント |
| `src/visualize_matcher/_visualize_matcher.py` | 可視化 |

---

## 結論

Issue #32の調査項目に対応するためには：

1. **FPFHの結果がRANSACに入れられるかどうか** → 既に入れられている ✅
2. **KDTreeパラメータの影響** → パラメータ化とベンチマークが必要 🔧
3. **fitnessの差** → 比較実験が必要 🔧
4. **処理速度の差** → 計時機能の追加が必要 🔧

次のステップとして、ベンチマークフレームワークの実装を推奨します。
