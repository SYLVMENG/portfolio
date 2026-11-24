from django.test import TestCase


class IndexPageTests(TestCase):
    def test_index_shows_results(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Choice probability")
