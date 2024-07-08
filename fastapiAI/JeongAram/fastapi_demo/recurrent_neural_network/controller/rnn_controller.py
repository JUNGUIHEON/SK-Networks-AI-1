from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from recurrent_neural_network.service.rnn_service_impl import RecurrentNeuralNetworkServiceImpl

RecurrentNeuralNetworkRouter = APIRouter()
# 흐름만 보는 라이트 버전입니다.
async def injectRecurrentNeuralNetworkService() -> RecurrentNeuralNetworkServiceImpl:
        return RecurrentNeuralNetworkServiceImpl()

@RecurrentNeuralNetworkRouter.post("/rnn-train")
async def rnnBasedTextTrain(recurrentNeuralNetworkService: RecurrentNeuralNetworkServiceImpl =
                            Depends(injectRecurrentNeuralNetworkService)):

    print(f"controller -> rnnBasedTrainText()")

    recurrentNeuralNetworkService.trainText()

class RnnRequestForm(BaseModel):
    inputText: str

@RecurrentNeuralNetworkRouter.post("/rnn-predict")
async def rnnBasedTextPredict(rnnRequestForm: RnnRequestForm,
                                  recurrentNeuralNetworkService: RecurrentNeuralNetworkServiceImpl =
                                  Depends(injectRecurrentNeuralNetworkService)):

    # print(f"rnnRequestForm: {rnnRequestForm}")
    inputText = rnnRequestForm.inputText

    if not inputText:
        raise HTTPException(status_code=400, detail='텍스트 입력을 해주세요!')

    print(f"controller -> rnnBasedTextPredict()")

    # 현재 상황에선 매우 형편 없을 것으로 기대됨
    generatedText = recurrentNeuralNetworkService.predictText(inputText)
    return { "generatedText": generatedText }


