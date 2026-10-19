"""Independent FK/IK regression for per-joint calibration and unchanged bounds."""
import unittest
import numpy as np
from advanced import Kinematics,load_advanced


class JointCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.model=Kinematics(load_advanced())
        self.raw=np.array([990,920,300,740,80,276])
        self.mapping=dict(raw_reference_unwrapped=self.model.unwrap(self.raw)[:5].tolist(),
            angle_reference_deg=np.degrees(self.model.angles(self.raw)+[.1,-.1,.05,.2,-.1]).tolist(),
            signed_degrees_per_raw=[.31,-.32,-.28,.34,.30])

    def test_individual_rates_and_roundtrip(self):
        adapted=self.model.with_joint_calibration(self.mapping)
        before=self.model.reference.copy()
        delta=np.array([2.,-3.,4.,-5.,6.,0.])
        expected=np.radians(np.array(self.mapping['angle_reference_deg'])+delta[:5]*self.mapping['signed_degrees_per_raw'])
        np.testing.assert_allclose(adapted.angles(self.raw+delta),expected)
        np.testing.assert_array_equal(self.model.reference,before)
        np.testing.assert_array_equal(abs(adapted.signs),np.ones(5))
        solved=adapted.inverse(adapted.forward(self.raw),self.raw,allow_crossing=False)
        np.testing.assert_allclose(adapted.forward(solved),adapted.forward(self.raw),atol=2)

    def test_raw_and_urdf_physical_limits_unchanged(self):
        adapted=self.model.with_joint_calibration(self.mapping)
        np.testing.assert_array_equal(adapted.limits,self.model.limits)
        for raw in np.random.default_rng(42).integers(0,1024,(100,6)):
            a=self.model.angles(raw);b=adapted.angles(raw)
            np.testing.assert_array_equal((a>=self.model.angle_limits[:,0])&(a<=self.model.angle_limits[:,1]),
                                          (b>=adapted.angle_limits[:,0])&(b<=adapted.angle_limits[:,1]))

    def test_invalid_calibration_rejected(self):
        broken=dict(self.mapping,signed_degrees_per_raw=[0.,1.,1.,1.,1.])
        with self.assertRaises(ValueError):self.model.with_joint_calibration(broken)


if __name__=='__main__':unittest.main()
