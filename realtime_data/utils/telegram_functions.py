
##? https://api.telegram.org/bot1997635645:AAGx_NugrgHZbGaoueLsR0yR3itpFhWznKw/getUpdates  for geting chat_ID
##? !pip install python-telegram-bot &> /dev/null
##? https://www.emojiall.com/en/sub-categories/J1  emoji
"""

python-telegram-bot==13.15

"""

import os
import telegram
import time
from datetime import datetime
import pytz


def start_telegram_bot(bot_token,chat_id,history,PRINT=True,TELEGRAM=True):
    now = datetime.now(pytz.timezone('Asia/Tehran'))
    date_string = now.strftime("%d/%m/%Y %H:%M:%S")
    bot = log_agent(bot_token=bot_token,chat_id=chat_id, PRINT=PRINT, TELEGRAM=TELEGRAM)
    bot.send_message(text="🚩"*15)
    bot.send_message(text="👇"*15)
    bot.send_message(text=f"bot_token ={bot_token}")
    bot.send_message(text=f"start date_time ={date_string}")    
    history['start_time'] = date_string
    history['bot_token'] = bot_token
    return bot,history

class log_agent():
    def __init__(self, bot_token, chat_id, PRINT,TELEGRAM):
        self.bot_token = bot_token
        self.bot = telegram.Bot(token=bot_token)
        self.chat_id = chat_id
        self.temp = {}
        print('telegram_agent initiated')
        
    def send_message(self,text,reply_to_message_id=None,PRINT=True,TELEGRAM=True):

        res = None
        if TELEGRAM:
            try:
                res = self.bot.send_message(text=text, chat_id=self.chat_id,reply_to_message_id = reply_to_message_id,timeout=50)
            except Exception as e:
                print("!!! telegram bot error")
                print(e)
        if PRINT:
            print(text)
        return res
    def send_photo(self, photo_binery=None, photo_address=None):
        
        try:
            if photo_binery: 
                res = self.bot.send_photo(chat_id=self.chat_id, photo=photo_binery, caption=None)
                return res
            elif photo_address:
                photo = open(photo_address,'rb')
                res = self.bot.send_photo(chat_id=self.chat_id, photo=photo, caption=None)
                return res
            else:
                print('! you must input image file or address.')
        except Exception as e:
            print("!!! telegram bot error")
            print(e)

    def forward_message(self,from_chat_id=None, message_id=None):
        res = self.bot.forward_message(chat_id=self.chat_id, from_chat_id=from_chat_id, message_id=message_id)
        return res

    def send_document(self, document=None, filename=None, caption=None, mode='main_bot'):
        res = None
        try:
            res = self.bot.send_document(chat_id=self.chat_id, document=document, filename=filename, caption=caption,timeout=150)
            return res
        except Exception as e:
            print("!!! telegram bot error")
            print(e)
        
        return res

    def send_file(self,file_address=None,CHUNK_SIZE=19000000,file_name='myfile',file_format ='.tar'):
        
        try:
            self.send_message(text= f"start uploading ...")
            self.temp = {'model_files':[]}
            os.makedirs('./splited_files_for_send/', exist_ok = True)
            file_number = 1
            with open(file_address,'rb') as f:
                chunk = f.read(CHUNK_SIZE)
                while chunk:
                    with open(f'./splited_files_for_send/{file_name}_{file_number}.{file_format}','wb') as chunk_file:
                        chunk_file.write(chunk)
                    with open(f'./splited_files_for_send/{file_name}_{file_number}.{file_format}','rb') as chunk_file:
                        res = self.send_document(document=chunk_file)  
                        self.send_message(text = str(res['document']['file_id']),reply_to_message_id=res['message_id'],PRINT=False)
                        self.temp['model_files'].append((res['document']['file_name'],res['document']['file_id']))
                    file_number += 1
                    chunk = f.read(CHUNK_SIZE)
            return self.temp
        except Exception as e:
            print("!!! telegram bot error")
            print(e)
            return None
