import unittest

from wan_va.distributed.fsdp import resolve_fsdp_topology


class FSDPTopologyTest(unittest.TestCase):
    def test_zero_preserves_whole_world_fsdp(self):
        topology = resolve_fsdp_topology(world_size=3, shard_size=0)
        self.assertEqual(topology.shard_size, 3)
        self.assertEqual(topology.replicate_size, 1)
        self.assertEqual(topology.mesh_shape, (3,))
        self.assertFalse(topology.uses_hsdp)

    def test_four_rank_two_way_hsdp(self):
        topology = resolve_fsdp_topology(world_size=4, shard_size=2)
        self.assertEqual(topology.shard_size, 2)
        self.assertEqual(topology.replicate_size, 2)
        self.assertEqual(topology.mesh_shape, (2, 2))
        self.assertTrue(topology.uses_hsdp)

    def test_rejects_uneven_or_invalid_groups(self):
        for world_size, shard_size in ((4, 3), (4, 5), (4, -1)):
            with self.subTest(world_size=world_size, shard_size=shard_size):
                with self.assertRaises(ValueError):
                    resolve_fsdp_topology(world_size, shard_size)


if __name__ == "__main__":
    unittest.main()
