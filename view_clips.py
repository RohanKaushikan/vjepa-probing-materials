import numpy as np, cv2
d = np.load("data/generated/elasticity/dataset.npz")

for i in [0, 50, 100, 200, 300, 400, 499]:
    print(f"video {i}: elasticity={d['elasticity_values'][i]:.4f}, seed={d['seeds'][i]}")
    for f in d["frames"][i]:
        cv2.imshow("video", f)
        cv2.waitKey(30)

cv2.destroyAllWindows()