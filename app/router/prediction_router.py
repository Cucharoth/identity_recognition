from app.exceptions.validation_exception import ValidationError
from app.services.classifier_service import ClassifierService
from app.services.validator_service import ValidatorService
from app.utils.response_builder import ResponseBuilder
from fastapi import APIRouter, HTTPException, UploadFile

from app.dto.api_response import ApiResponse
from app.utils.logger import Logger 

router_logger = Logger()

prediction_router = APIRouter()

@prediction_router.post("/verify", response_model=ApiResponse)
def verify(file: UploadFile):
   try:
      router_logger.info('[Prediction] Verify endpoint called')

      ValidatorService().validate(file)
      result = ClassifierService().verify(file)

      return ResponseBuilder.success(result, model_version=result["model_version"])
   except ValidationError as e:
      router_logger.error(f'[Prediction] Validation error: {str(e)}')
      raise HTTPException(status_code=400, detail=ResponseBuilder.error(str(e), code=400))
   except Exception as e:
      router_logger.error(f'[Prediction] Error processing query: {str(e)}')
      raise HTTPException(
         status_code=500, 
         detail=ResponseBuilder.error("Lo sentimos, ha ocurrido un error procesando tu consulta.", code=500)
      )