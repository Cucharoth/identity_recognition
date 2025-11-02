from app.exceptions.no_face_exception import NoFaceDetectedError
from app.exceptions.validation_exception import ValidationError
from app.services.classifier_service import ClassifierService
from app.services.validator_service import ValidatorService
from app.utils.response_builder import ResponseBuilder
from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.dto.api_response import ApiResponse
from app.utils.logger import Logger 

router_logger = Logger()

prediction_router = APIRouter()


def get_validator_service() -> ValidatorService:
   return ValidatorService()


def get_classifier_service() -> ClassifierService:
   return ClassifierService()


@prediction_router.post("/verify", response_model=ApiResponse)
def verify(
   file: UploadFile,
   validator: ValidatorService = Depends(get_validator_service),
   classifier: ClassifierService = Depends(get_classifier_service),
):
   try:
      router_logger.info('[Verify] Verify endpoint called')

      validator.validate(file)
      result = classifier.verify(file)

      return ResponseBuilder.success(result)
   except ValidationError as e:
      router_logger.error(f'[Verify] Validation error: {str(e)}')
      raise HTTPException(status_code=400, detail=ResponseBuilder.error(str(e), code=400))
   except NoFaceDetectedError as e:
      router_logger.error(f'[Verify] No face detected: {str(e)}')
      raise HTTPException(status_code=422, detail=ResponseBuilder.error("No se detectó ningún rostro en la imagen.", code=422))
   except Exception as e:
      router_logger.error(f'[Verify] Error processing query: {str(e)}')
      raise HTTPException(
         status_code=500, 
         detail=ResponseBuilder.error("Lo sentimos, ha ocurrido un error procesando tu consulta.", code=500)
      )