from unittest import TestCase

from project.priority_queue import PriorityQueue


class TestPriorityQueue(TestCase):
    def test_pop(self):
        pq = PriorityQueue()
        pq += (2, 3)
        pq += (1, 2)
        pq += (0, 1)
        popped = pq.pop()

        self.assertEqual((0, 1), popped)

    def test_remove(self):
        pq = PriorityQueue()
        pq += (2, 3)
        pq += (1, 2)
        pq += (0, 1)
        pq.remove((0, 1))
        pq.remove((1, 2))

        popped = pq.pop()

        self.assertEqual((2, 3), popped)

    def test_push(self):
        pq = PriorityQueue()
        pq += (2, 3)
        pq += (1, 2)
        pq += (0, 1)

        popped = pq.pop()
        self.assertEqual((0, 1), popped)

        popped = pq.pop()
        self.assertEqual((1, 2), popped)

        popped = pq.pop()
        self.assertEqual((2, 3), popped)

    def test_len(self):
        pq = PriorityQueue()
        pq += (2, 3)
        pq += (1, 2)
        pq += (0, 1)

        self.assertEqual(3, len(pq))

        pq.pop()
        self.assertEqual(2, len(pq))

        pq.pop()
        self.assertEqual(1, len(pq))

        pq.pop()
        self.assertEqual(0, len(pq))
