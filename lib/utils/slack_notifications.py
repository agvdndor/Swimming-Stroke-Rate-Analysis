import os
from slack import WebClient 
from secrets import oauth_acces_token, bot_user_oauth_acces_token

# acces tokens for swimthesis.slack.com
#post message in channel

def slack_message(message, channel):
    token = bot_user_oauth_acces_token
    sc = WebClient(token)
    sc.chat_postMessage(channel=channel, 
                text=message, username='Fake Human',
                icon_emoji=':computer:')

if __name__ == "__main__":
    # test
    slack_message('test message', '#random')