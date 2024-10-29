import requests
import json
import time
from configs.discord_settings import discord_url

"""
 docs for discord reporter:

 https://pypi.org/project/discordwebhook/
 https://github.com/lovvskillz/python-discord-webhook
 https://birdie0.github.io/discord-webhooks-guide/structure/username.html
 https://discord.com/developers/docs/resources/webhook#create-webhook
 https://discord.com/developers/docs/resources/channel#embed-object

"""

COLORS = {
    "red": 16711680,
    "orange": 16743168,
    "yellow": 16772608,
    "green": 5373696,
    "blue": 36095,
    "pink": 14221567,
}


class DiscordWebhook:
    def __init__(self, task_name, channel_name, mode="online"):
        """
        mode: "online" , "offline"
        
        """
        self.channel_webhook = discord_url[channel_name]
        self.task_name = task_name
        self.mode = mode
        self.t0 = time.time()
        self.headers = {"Content-Type": "application/json"}
        self.start_massage()

    def __repr__(self):
        return self.task_name

    def do_request(self, data=None, file=None, try_count=5):
        counter = 0
        while True:
            try:
                time.sleep(2)
                result = requests.post(
                    self.channel_webhook, json=data, files=file, timeout=35
                )
                if 200 <= result.status_code < 300:
                    pass
                    # print(f"Webhook sent {result.status_code}")
                else:
                    print(
                        f"Not sent with {result.status_code}, response:\n{result.json()}"
                    )
                break
            except Exception as e:
                print(f"error in sending discord message: {self.task_name}")
                print(e)
                counter += 1
                if counter >= try_count:
                    break
                time.sleep(5)

    def send_dict_as_text_file(
        self, input_dict, massage=None, file_name="report_dict.text"
    ):
        data = {
            "username": f"task: {self.task_name} Report:",
            "content": f"{massage}",
        }
        files = {file_name: (file_name, json.dumps(input_dict))}
        files["payload_json"] = (None, json.dumps(data))
        self.do_request(file=files, try_count=5)

    def send_list_as_text_file(
        self, input_list, massage=None, file_name="report_list.text"
    ):
        data = {
            "username": f"task: {self.task_name} Report:",
            "content": f"{massage}",
        }
        string_bag = ""
        for line in input_list:
            string_bag += str(line)
            string_bag += "\n"
        files = {file_name: (file_name, string_bag)}
        files["payload_json"] = (None, json.dumps(data))
        self.do_request(file=files, try_count=5)

    def send_message_with_embed(self, massage, description="", color=COLORS["blue"]):
        task_duration = f"| {((time.time() - self.t0) / 60):.2f} minutes passed | "
        massage = str(massage)
        massage = task_duration + massage
        
        print(massage)

        if self.mode == "online":
            embed = {
                "title": f"{self.task_name} Report:",
                "description": description,
                "color": color,
            }
            data = {
                "username": f"task: {self.task_name} Report:",
                "content": f"{massage}",
                "embeds": [embed],
            }
            self.do_request(data, try_count=5)

    def send_message(self, massage):
        task_duration = f"| {((time.time() - self.t0) / 60):.2f} minutes passed | "
        massage = str(massage)
        massage = task_duration + massage
        print(massage)
        if self.mode == "online":
            data = {
                "username": f"task: {self.task_name} Report:",
                "content": f"{massage}",
            }
            self.do_request(data, try_count=5)

    def start_massage(self):
        massage = ""
        color = COLORS["green"]
        self.send_message_with_embed(massage, color=color)


def send_sample_message():
    discord_reporter = DiscordWebhook("sample name of a task")
    discord_reporter.send_message("a good result")
    discord_reporter.send_message("403 error")
