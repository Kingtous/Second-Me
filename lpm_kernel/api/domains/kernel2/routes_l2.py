import json
import logging
import os
import time
from dataclasses import asdict

from flask import Blueprint, jsonify, Response, request
from flask_pydantic import validate

from lpm_kernel.L1.serializers import NotesStorage
from lpm_kernel.L1.utils import save_true_topics
from lpm_kernel.L2.l2_generator import L2Generator
from lpm_kernel.L2.utils import save_hf_model
from lpm_kernel.api.common.responses import APIResponse
from lpm_kernel.api.domains.kernel2.dto.chat_dto import (
    ChatRequest,
)
from lpm_kernel.api.domains.kernel2.services.chat_service import chat_service
from lpm_kernel.api.domains.kernel2.services.prompt_builder import (
    BasePromptStrategy,
    RoleBasedStrategy,
    KnowledgeEnhancedStrategy,
)
from lpm_kernel.api.domains.kernel2.services.role_service import role_service
from lpm_kernel.api.domains.loads.services import LoadService
from lpm_kernel.api.services.local_llm_service import local_llm_service
from lpm_kernel.kernel.chunk_service import ChunkService
from lpm_kernel.kernel.l1.l1_manager import (
    extract_notes_from_documents,
    document_service,
    get_latest_status_bio,
    get_latest_global_bio,
)
from ...common.script_executor import ScriptExecutor
from ...common.script_runner import ScriptRunner
from ....configs.config import Config
from ....kernel.note_service import NoteService

logger = logging.getLogger(__name__)

kernel2_bp = Blueprint("kernel2", __name__, url_prefix="/api/kernel2")

# Create script executor instance
script_executor = ScriptExecutor()


@kernel2_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    config = Config.from_env()
    app_name = config.app_name or "Service"  # Add default value to prevent None

    status = local_llm_service.get_server_status()
    if status.is_running and status.process_info:
        return jsonify(
            APIResponse.success(
                data={
                    "status": "running",
                    "pid": status.process_info.pid,
                    "cpu_percent": status.process_info.cpu_percent,
                    "memory_percent": status.process_info.memory_percent,
                    "uptime": time.time() - status.process_info.create_time,
                }
            )
        )
    else:
        return jsonify(APIResponse.success(data={"status": "stopped"}))


@kernel2_bp.route("/model/download", methods=["POST"])
def downloadModel():
    """Download base model
    
    Request body:
    {
        "model_name": str  # Model name, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    }
    
    Returns:
    {
        "code": int,
        "message": str,
        "data": {
            "model_path": str  # Model save path
        }
    }
    """
    try:
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]

        # Download and save model
        model_path = save_hf_model(model_name)

        return jsonify(APIResponse.success(
            data={"model_path": model_path},
            message="Model download completed"
        ))

    except Exception as e:
        error_msg = f"Failed to download model: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


@kernel2_bp.route("/username", methods=["GET"])
def username():
    return jsonify(APIResponse.success(data={"username": LoadService.get_current_upload_name()}))

# read IN_DOCKER_ENV and output
@kernel2_bp.route("/docker/env", methods=["GET"])
def docker_env():
    return jsonify(APIResponse.success(data={"in_docker_env": os.getenv("IN_DOCKER_ENV")}))
    


@kernel2_bp.route("/data/prepare", methods=["POST"])
def all():
    def generate():
        try:
            # 1. Initialize configuration and directories (5%)
            progress_data = {
                "stage": "Initializing",
                "progress": 5,
                "message": "Initializing configuration and directories"
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            config = Config.from_env()
            base_dir = os.path.join(
                os.getcwd(), config.get("USER_DATA_PIPELINE_DIR") + "/raw_data"
            )
            os.makedirs(base_dir, exist_ok=True)

            # 2. Process topics data (15%)
            progress_data = {
                "stage": "Processing Topics",
                "progress": 15,
                "message": "Saving topics data"
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            chunk_service = ChunkService()
            topics_data = chunk_service.query_topics_data()
            save_true_topics(topics_data, os.path.join(base_dir, "topics.json"))

            # 3. Process documents and notes (35%)
            progress_data = {
                "stage": "Processing Documents",
                "progress": 35,
                "message": "Extracting and preparing document notes"
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            documents = document_service.list_documents_with_l0()
            notes_list, _ = extract_notes_from_documents(documents)
            if not notes_list:
                error_data = {
                    "stage": "Error",
                    "progress": -1,
                    "message": "No notes found"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            note_service = NoteService()
            note_service.prepareNotes(notes_list)

            storage = NotesStorage()
            result = storage.save_notes(notes_list)

            # 4. Prepare configuration files and paths (50%)
            progress_data = {
                "stage": "Preparing Configuration",
                "progress": 50,
                "message": "Preparing L2 generator configuration "
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            config_path = os.path.join(
                os.getcwd(),
                "resources/L2/data_pipeline/data_prep/subjective/config/config.json",
            )
            entitys_path = os.path.join(
                os.getcwd(),
                "resources/L2/data_pipeline/raw_data/id_entity_mapping_subjective_v2.json",
            )
            graph_path = os.path.join(
                os.getcwd(),
                "resources/L1/graphrag_indexing_output/subjective/entities.parquet",
            )

            data_output_base_dir = os.path.join(os.getcwd(), "resources/L2/data")
            notes = storage.load_notes()

            # 5. Prepare basic information (65%)
            progress_data = {
                "stage": "Preparing Basic Info",
                "progress": 65,
                "message": "Getting user information and bio"
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            status_bio = get_latest_status_bio()
            global_bio = get_latest_global_bio()

            basic_info = {
                "username": LoadService.get_current_upload_name(),
                "aboutMe": LoadService.get_current_upload_description(),
                "statusBio": status_bio.content
                if status_bio
                else "Currently working on an AI project.",
                "globalBio": global_bio.content_third_view
                if global_bio
                else "The User is a software engineer who loves programming and learning new technologies.",
                "lang": "English",
            }

            # 6. Data preprocessing (80%)
            progress_data = {
                "stage": "Data Preprocessing",
                "progress": 80,
                "message": "Executing data preprocessing"
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            l2_generator = L2Generator(
                data_path=os.path.join(os.getcwd(), "resources")
            )
            l2_generator.data_preprocess(notes, basic_info)

            # 7. Generate subjective data (95%)
            progress_data = {
                "stage": "Generating Data",
                "progress": 95,
                "message": "Generating subjective data： Preference QA Self QA Diversity QA graphrag_indexing"
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

            l2_generator.gen_subjective_data(
                notes,
                basic_info,
                data_output_base_dir,
                storage.topics_path,
                entitys_path,
                graph_path,
                config_path,
            )

            # 8. Complete (100%)
            progress_data = {
                "stage": "Complete",
                "progress": 100,
                "message": "Data preparation completed",
                "result": {
                    "bio": basic_info["globalBio"],
                    "document_clusters": "Generated document clusters",
                    "chunk_topics": "Generated chunk topics"
                }
            }
            yield f"data: {json.dumps(progress_data)}\n\n"

        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}", exc_info=True)
            error_data = {
                "stage": "Error",
                "progress": -1,
                "message": f"Data preparation failed: {str(e)}"
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # Disable Nginx buffering
        }
    )


# Global variables for tracking training process
_training_process = None
_training_thread = None
_stopping_training = False


def get_model_paths(model_name: str) -> dict:
    """
    Get all paths related to the model
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary containing all related paths:
        - base_path: Base model path
        - personal_dir: Personal trained model output directory
        - merged_dir: Merged model output directory
    """
    base_dir = os.getcwd()
    paths = {
        "base_path": os.path.join(base_dir, "resources/L2/base_models", model_name),
        "personal_dir": os.path.join(base_dir, "resources/model/output/personal_model", model_name),
        "merged_dir": os.path.join(base_dir, "resources/model/output/merged_model", model_name),
        "gguf_dir": os.path.join(base_dir, "resources/model/output/gguf", model_name)
    }

    # Ensure all directories exist
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def start_training(script_path: str, log_path: str) -> None:
    """Start training in a new thread"""
    global _training_process
    try:
        # Use ScriptRunner to execute the script
        runner = ScriptRunner(log_path=log_path)
        _training_process = runner.execute_script(
            script_path=script_path,
            script_type="training",
            is_python=False,  # This is a bash script
        )

        logger.info(f"Training process started: {_training_process}")

    except Exception as e:
        logger.error(f"Failed to start training process: {str(e)}")
        _training_process = None
        raise


@kernel2_bp.route("/train2", methods=["POST"])
def train2():
    """Start model training"""
    global _training_thread, _training_process, _stopping_training

    try:
        # Get request parameters
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]
        paths = get_model_paths(model_name)

        # Check if model exists
        if not os.path.exists(paths["base_path"]):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' does not exist, please download first",
                code=400
            ))

        # Check if training is already running
        if _training_thread and _training_thread.is_alive():
            return jsonify(APIResponse.error("Training task is already running"))

        # Reset stopping flag
        _stopping_training = False

        # Prepare log directory and file
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "train.log")
        logger.info(f"Log file path: {log_path}")

        # Ensure output directory exists
        os.makedirs(paths["personal_dir"], exist_ok=True)

        # Set environment variables
        os.environ["MODEL_BASE_PATH"] = paths["base_path"]
        os.environ["MODEL_PERSONAL_DIR"] = paths["personal_dir"]
        # Assign
        os.environ["USER_NAME"] = LoadService.get_current_upload_name()

        logger.info(f"Environment variables set: {os.environ}")

        script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/train_for_user.sh")

        # Start training
        import threading
        _training_thread = threading.Thread(
            target=start_training,
            args=(script_path, log_path),
            daemon=True
        )
        _training_thread.start()

        return jsonify(APIResponse.success(
            data={
                "model_name": model_name,
                "log_path": log_path,
                "personal_dir": paths["personal_dir"],
                "merged_dir": paths["merged_dir"]
            },
            message="Training task started"
        ))

    except Exception as e:
        error_msg = f"Failed to start training: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


@kernel2_bp.route("/merge_weights", methods=["POST"])
def merge_weights():
    """Merge model weights"""
    try:
        # Get request parameters
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]
        paths = get_model_paths(model_name)

        # Check if model exists
        if not os.path.exists(paths["base_path"]):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' does not exist, please download first",
                code=400
            ))

        # Check if training output exists
        if not os.path.exists(paths["personal_dir"]):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' training output does not exist, please train model first",
                code=400
            ))

        # Ensure merged output directory exists
        os.makedirs(paths["merged_dir"], exist_ok=True)

        # Set environment variables
        os.environ["MODEL_BASE_PATH"] = paths["base_path"]
        os.environ["MODEL_PERSONAL_DIR"] = paths["personal_dir"]
        os.environ["MODEL_MERGED_DIR"] = paths["merged_dir"]

        logger.info(f"Environment variables set: MODEL_BASE_PATH :  {os.environ}")

        script_path = os.path.join(
            os.getcwd(), "lpm_kernel/L2/merge_weights_for_user.sh"
        )
        log_path = os.path.join(os.getcwd(), "logs", f"merge_weights_{model_name}.log")

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Use script executor to execute merge script
        result = script_executor.execute(
            script_path=script_path, script_type="merge_weights", log_file=log_path
        )

        return jsonify(
            APIResponse.success(
                data={
                    **result,
                    "model_name": model_name,
                    "log_path": log_path,
                    "personal_dir": paths["personal_dir"],
                    "merged_dir": paths["merged_dir"]
                },
                message="Weight merge task started"
            )
        )

    except Exception as e:
        error_msg = f"Failed to start weight merge: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


@kernel2_bp.route("/convert_model", methods=["POST"])
def convert_model():
    """Convert model to GGUF format"""
    try:
        # Get request parameters
        data = request.get_json()
        logger.info(f"Request parameters: {data}")
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]
        logger.info(f"Converting model: {model_name}")
        paths = get_model_paths(model_name)

        # Check if merged model exists
        merged_model_dir = paths["merged_dir"]
        logger.info(f"Merged model path: {merged_model_dir}")
        if not os.path.exists(merged_model_dir):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' merged output does not exist, please merge model first",
                code=400
            ))

        # Get GGUF output directory
        gguf_dir = paths["gguf_dir"]
        logger.info(f"GGUF output directory: {gguf_dir}")

        script_path = os.path.join(os.getcwd(), "lpm_kernel/L2/convert_hf_to_gguf.py")
        gguf_path = os.path.join(gguf_dir, "model.gguf")
        logger.info(f"GGUF output path: {gguf_path}")

        # Build parameters
        args = [
            merged_model_dir,
            "--outfile",
            gguf_path,
            "--outtype",
            "f16",
        ]
        logger.info(f"Parameters: {args}")
        # Use script executor to execute conversion script
        result = script_executor.execute(
            script_path=script_path, script_type="convert_model", args=args
        )

        logger.info(f"Model conversion successful: {result}")
        return jsonify(APIResponse.success(
            data={
                **result,
                "model_name": model_name,
                "merged_dir": merged_model_dir,
                "gguf_path": gguf_path
            },
            message="Model conversion task started"
        ))

    except Exception as e:
        error_msg = f"Failed to start model conversion: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


@kernel2_bp.route("/llama/start", methods=["POST"])
def start_llama_server():
    """Start llama-server service"""
    try:
        # Get request parameters
        data = request.get_json()
        if not data or "model_name" not in data:
            return jsonify(APIResponse.error(message="Missing required parameter: model_name", code=400))

        model_name = data["model_name"]
        paths = get_model_paths(model_name)
        gguf_path = os.path.join(paths["gguf_dir"], "model.gguf")
        server_path = os.path.join(os.getcwd(), "llama.cpp/llama-server.exe")

        # Check if service and model file exist
        if not os.path.exists(server_path):
            return jsonify(APIResponse.error(message="llama-server executable file does not exist", code=400))
        if not os.path.exists(gguf_path):
            return jsonify(APIResponse.error(
                message=f"Model '{model_name}' GGUF file does not exist, please convert model first",
                code=400
            ))

        # Check if service is already running
        status = local_llm_service.get_server_status()
        if status.is_running:
            return jsonify(
                APIResponse.error(
                    message=f"llama-server is already running, PID: {status.process_info.pid}",
                    code=400
                )
            )

        # Build parameters
        args = [server_path, "-m", gguf_path, "--port", "8080"]

        # Use thread to start service asynchronously
        def start_server():
            script_executor.execute(
                script_path=server_path,
                script_type="llama_server",
                args=args[1:],  # Remove first parameter (executable file path)
                shell=False,
            )

        # Start new thread to run service
        from threading import Thread

        thread = Thread(target=start_server)
        thread.daemon = True
        thread.start()

        # Return start status immediately
        return jsonify(
            APIResponse.success(
                data={
                    "model_name": model_name,
                    "gguf_path": gguf_path,
                    "status": "starting"
                },
                message="llama-server service is starting"
            )
        )

    except Exception as e:
        error_msg = f"Failed to start llama-server: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(message=error_msg, code=500))


# Flag to track if service is stopping
_stopping_server = False


@kernel2_bp.route("/llama/stop", methods=["POST"])
def stop_llama_server():
    """Stop llama-server service - Force immediate termination of the process"""
    global _stopping_server

    try:
        # If service is already stopping, return notification
        if _stopping_server:
            return jsonify(APIResponse.success(message="llama-server service is stopping"))

        _stopping_server = True  # Set stopping flag

        try:
            # use improved local_llm_service.stop_server() to stop all llama-server process
            status = local_llm_service.stop_server()

            # check if there are still processes running
            if status.is_running and status.process_info:
                pid = status.process_info.pid
                logger.warning(f"llama-server process still running: {pid}")
                return jsonify(APIResponse.success(
                    message="llama-server service could not be fully stopped. Please try again.",
                    data={"running_pid": pid}
                ))
            else:
                return jsonify(APIResponse.success(message="llama-server service has been stopped successfully"))

        except Exception as e:
            logger.error(f"Error while stopping llama-server: {str(e)}")
            return jsonify(APIResponse.error(message=f"Error stopping llama-server: {str(e)}", code=500))
        finally:
            _stopping_server = False

    except Exception as e:
        _stopping_server = False
        logger.error(f"Failed to stop llama-server: {str(e)}")
        return jsonify(APIResponse.error(message=f"Failed to stop llama-server: {str(e)}", code=500))


@kernel2_bp.route("/llama/status", methods=["GET"])
@validate()
def get_llama_server_status():
    """Get llama-server service status"""
    try:
        status = local_llm_service.get_server_status()
        return APIResponse.success(asdict(status))

    except Exception as e:
        logger.error(f"Error getting llama-server status: {str(e)}", exc_info=True)
        return APIResponse.error(f"Error getting llama-server status: {str(e)}")


@kernel2_bp.route("/test/version", methods=["GET"])
def test_version():
    """Test environment version"""
    try:
        # Execute python command directly to get version
        result = script_executor.execute(
            script_path="python", script_type="version_check", args=["--version"]
        )

        return jsonify(
            APIResponse.success(
                data={"python_version": result}, message="Version information obtained successfully"
            )
        )

    except Exception as e:
        error_msg = f"Failed to get version information: {str(e)}"
        logger.error(error_msg)
        return jsonify(APIResponse.error(error_msg))


@kernel2_bp.route("/chat", methods=["POST"])
@validate()
def chat(body: ChatRequest):
    """
    Chat interface - Stream response (OpenAI API compatible)

    Request parameters: Compatible with OpenAI Chat Completions API format
    - messages: List[Dict[str, str]], standard OpenAI message list with format:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I am a helpful assistant."},
            {"role": "user", "content": "What can you do for me?"}  
        ]
    - metadata: Dict[str, Any], additional parameters for request processing (optional):
        {
            "enable_l0_retrieval": true,  // whether to enable knowledge retrieval
            "enable_l1_retrieval": false, // whether to enable advanced knowledge retrieval
            "role_id": "uuid-string"      // optional role UUID for system customization
        }
    - stream: bool, whether to stream the response (default: True)
    - model: str, model identifier (optional, default uses configured model)
    - temperature: float, controls randomness (default: 0.1)
    - max_tokens: int, maximum tokens to generate (default: 2000)

    Response: Standard OpenAI Chat Completions API format
    For stream=true (Server-Sent Events):
    - id: str, response unique identifier
    - object: "chat.completion.chunk"
    - created: int, timestamp
    - model: str, model identifier
    - system_fingerprint: str, system fingerprint
    - choices: [
        {
          "index": 0,
          "delta": {"content": str},
          "finish_reason": null or "stop"
        }
      ]
    
    The last event will be: data: [DONE]
    
    For stream=false:
    - Complete response object with full message content
    """
    try:
        logger.info(f"Starting chat request: {body}")
        # 1. Check service status
        status = local_llm_service.get_server_status()
        if not status.is_running:
            # Format error response in OpenAI-compatible format
            error_msg = "LLama server is not running"
            logger.error(error_msg)
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "server_error",
                    "code": "service_unavailable"
                }
            }
            # Return as regular JSON response for non-stream or stream-compatible error
            if not body.stream:
                return APIResponse.error(message="服务暂时不可用", code=503), 503
            return local_llm_service.handle_stream_response(iter([error_response]))

        try:
            # Use chat_service to process request with OpenAI-compatible format
            response = chat_service.chat(
                request=body,
                stream=body.stream,  # Respect the stream parameter from request
                json_response=False,
                strategy_chain=[BasePromptStrategy, RoleBasedStrategy, KnowledgeEnhancedStrategy]
            )
            
            # Handle streaming or non-streaming response appropriately
            if body.stream:
                return local_llm_service.handle_stream_response(response)
            else:
                # For non-streaming, return the complete response as JSON
                return jsonify(response)

        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Value error: {error_msg}")
            error_response = {
                "error": {
                    "message": error_msg,
                    "type": "invalid_request_error",
                    "code": "bad_request"
                }
            }
            if not body.stream:
                return jsonify(error_response), 400
            return local_llm_service.handle_stream_response(iter([error_response]))

    except Exception as e:
        error_msg = f"Request processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        error_response = {
            "error": {
                "message": error_msg,
                "type": "server_error",
                "code": "internal_server_error"
            }
        }
        if not getattr(body, 'stream', True):  # Default to stream if attribute missing
            return jsonify(error_response), 500
        return local_llm_service.handle_stream_response(iter([error_response]))
