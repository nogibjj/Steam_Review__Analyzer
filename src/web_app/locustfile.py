from locust import HttpUser, task


class WebsiteUser(HttpUser):
    @task
    def load_the_test(self):
        self.client.get("/")
