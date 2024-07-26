from pydantic import BaseModel

# 유저가 전달하는 메시지
class SrbcbRequestForm(BaseModel):
    userSendMessage: str
