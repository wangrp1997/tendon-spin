"""CPU-consumer ordering and isolation at the audited compatibility boundary."""
from types import SimpleNamespace
import unittest

from tendonspin.physics.physx_property_sync import synchronized_copy
from scripts.resume_boya_spin_progress_milestones import remaining_targets


class PropertySyncTests(unittest.TestCase):
    def run_consumer(self, *, filename='/audited/rigid_object.py', function='set_masses_index', cpu=True, explicit_stream=None):
        stream=object();pending=[];waits=[];observed=[]
        dest=SimpleNamespace(device=SimpleNamespace(is_cpu=cpu),value=0)
        src=SimpleNamespace(device=SimpleNamespace(is_cuda=True,stream=stream),value=.05)

        def asynchronous_copy(dst,source,**kwargs):
            pending.append(lambda:setattr(dst,'value',source.value))

        def synchronize(actual_stream):
            self.assertIs(actual_stream,explicit_stream or stream)
            while pending:pending.pop(0)()

        copy=synchronized_copy(asynchronous_copy,synchronize,'/audited/rigid_object.py',waits.append)
        namespace=dict(copy=copy,dest=dest,src=src,observed=observed,explicit_stream=explicit_stream)
        exec(compile(f'def {function}():\n    copy(dest,src,stream=explicit_stream)\n    observed.append(dest.value)\n',filename,'exec'),namespace)
        namespace[function]()
        return observed,waits,pending

    def test_cpu_consumer_sees_completed_mass_copy(self):
        observed,waits,pending=self.run_consumer()
        self.assertEqual(observed,[.05]);self.assertEqual(waits,['set_masses_index']);self.assertFalse(pending)

    def test_com_and_ids_wait_on_explicit_copy_stream(self):
        for name in ('set_coms_index','_get_cpu_env_ids'):
            observed,waits,pending=self.run_consumer(function=name,explicit_stream=object())
            self.assertEqual(observed,[.05]);self.assertEqual(waits,[name]);self.assertFalse(pending)

    def test_other_callers_and_gpu_copies_remain_asynchronous(self):
        for kwargs in (dict(filename='/other/rigid_object.py'),dict(function='solver_step'),dict(cpu=False)):
            observed,waits,pending=self.run_consumer(**kwargs)
            self.assertEqual(observed,[0]);self.assertFalse(waits);self.assertTrue(pending)

    def test_recovery_does_not_repeat_rounded_completed_milestone(self):
        targets=[dict(requested_actions=10,rounded_actions=16),dict(requested_actions=20,rounded_actions=24),dict(requested_actions=30,rounded_actions=32)]
        self.assertEqual(remaining_targets(targets,24),targets[2:])
        self.assertEqual(remaining_targets(targets,32),[])
        with self.assertRaises(ValueError):remaining_targets(targets,20)


if __name__=='__main__':
    unittest.main()
