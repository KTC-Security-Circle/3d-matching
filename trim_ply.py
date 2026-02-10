import open3d as o3d


def manual_cropping(input_path, output_path):
    print("ウィンドウが開いたら以下の操作を行ってください：")
    print("1. [Shift + 左マウスドラッグ] で選択範囲を作成")
    print("2. [C] キーを押してクロップ（選択範囲内のみ残す）")
    print("3. 必要に応じて繰り返し")
    print("4. [Q] キーまたはESCキーでウィンドウを閉じる")

    pcd = o3d.io.read_point_cloud(input_path)
    print(f"\n入力ファイル: {input_path}")
    print(f"出力ファイル: {output_path}")
    print(f"点群データ読み込み完了: {len(pcd.points)} 点\n")

    # VisualizerWithEditingを使用して編集
    print("編集ウィンドウを開きます...")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # ユーザーが編集するまで待機
    vis.destroy_window()

    # 編集結果を取得
    cropped_pcd = vis.get_cropped_geometry()

    if cropped_pcd is not None and len(cropped_pcd.points) > 0:
        print(f"\n編集後の点数: {len(cropped_pcd.points)} 点")
        print(f"保存先: {output_path}")

        result = o3d.io.write_point_cloud(output_path, cropped_pcd)
        if result:
            print(f"✓ 編集結果を {output_path} に保存しました")
            print(f"  削減: {len(pcd.points)} → {len(cropped_pcd.points)} 点")
        else:
            print("✗ エラー: 保存に失敗しました")
    else:
        print("\n警告: クロップされた点群がありません")
        print("元のデータをそのまま保存します")
        o3d.io.write_point_cloud(output_path, pcd)


if __name__ == "__main__":
    manual_cropping("3d_data/target(scaned).ply", "3d_data/cleaned_data.ply")
