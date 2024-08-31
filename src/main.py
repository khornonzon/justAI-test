from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from pydantic import BaseModel
from NERClassifier import PERextractor
from typing import Dict, Any, List

id2label = {0: 'B-LOC', 1: 'B-ORG', 2: 'I-LOC', 3:'I-ORG', 4:'O', 5: 'PER'}

class PredictRequest(BaseModel):
    texts: List[str]

    def __init__(self, texts):
        super().__init__(texts=texts)


class PredictResponse(BaseModel):
    entities_list: List[Any]

    def __init__(self, entities_list):
        super().__init__(entities_list=entities_list)


class SimpleActionExample(Task):
    def __init__(self, config: BaseModel, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)
        self.model =  PERextractor('/app/model', "DeepPavlov/rubert-base-cased", id2label)
    def predict(self, data: PredictRequest, config: BaseModel) -> PredictResponse:
        input_data = {'texts':data.texts}
        output_data = self.model.predictionsFromDict(input_data)
    
        return PredictResponse(entities_list=output_data['entities_list'])


if __name__ == "__main__":
    host_mlp_cloud(SimpleActionExample, BaseModel())
