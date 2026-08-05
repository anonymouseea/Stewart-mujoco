import numpy as np
from IKFK import stewart_ik, stewart_fk

test_cases = [
    ([0.0, 0.0, 0.65], np.deg2rad([0.0, 0.0, 0.0])),
    ([0.01, -0.01, 0.62], np.deg2rad([2.0, -1.0, 3.0])),
    ([-0.02, 0.015, 0.68], np.deg2rad([-3.0, 2.0, -2.0])),
]

for pos, rpy in test_cases:
    pos = np.array(pos, dtype=float)

    lengths, _ = stewart_ik(pos, rpy)

    fk_pos, fk_rpy, info = stewart_fk(
        lengths,
        initial_pose=[0.0, 0.0, 0.65, 0.0, 0.0, 0.0],
        return_info=True,
    )

    print("target pos:", pos)
    print("fk pos:    ", fk_pos)
    print("pos error: ", np.linalg.norm(fk_pos - pos))

    print("target rpy deg:", np.rad2deg(rpy))
    print("fk rpy deg:    ", np.rad2deg(fk_rpy))
    print("rpy error deg: ", np.rad2deg(np.linalg.norm(fk_rpy - rpy)))

    print("info:", info)
    print()