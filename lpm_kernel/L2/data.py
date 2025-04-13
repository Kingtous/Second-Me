"""Data processing module for L2 model training.

This module provides the L2DataProcessor class which handles data preparation,
processing, and organization for training L2 models, including conversion between
different formats and extraction of information from notes.
"""

import graphrag
import json
import os
import pandas as pd
import random
import subprocess
import traceback
import yaml
from collections import defaultdict
from datasets import DatasetDict, Dataset
from datetime import datetime
from tqdm import tqdm
from typing import Any, Dict, List

from lpm_kernel.L1.bio import (
    MemoryType,
    Note,
    OBJECT_NOTE_TYPE,
    SUBJECT_NOTE_TYPE
)
from lpm_kernel.L2.data_pipeline.data_prep.context_data.context_generator import ContextGenerator
from lpm_kernel.L2.data_pipeline.data_prep.diversity.diversity_data_generator import DiversityDataGenerator
from lpm_kernel.L2.data_pipeline.data_prep.preference.preference_QA_generate import PreferenceQAGenerator
from lpm_kernel.L2.data_pipeline.data_prep.selfqa.selfqa_generator import SelfQA
from lpm_kernel.L2.note_templates import OBJECTIVE_TEMPLATES, SUBJECTIVE_TEMPLATES
from lpm_kernel.L2.utils import format_timestr
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

class L2DataProcessor:
    """Data processor for L2 model training.
    
    This class handles the processing and organization of data for training L2 models,
    including conversion between different formats, extraction of information from notes,
    and preparation of training data.
    
    Attributes:
        data_path: Base path for data processing.
        preferred_lang: Preferred language for data processing.
    """

    def __init__(
            self,
            data_path: str = "resources/L2/data_pipeline/raw_data",
            preferred_lang: str = "English",
    ):
        """Initialize the L2DataProcessor.
        
        Args:
            data_path: Base path for data processing. Defaults to "resources/L2/data_pipeline/raw_data".
            preferred_lang: Preferred language for data processing. Defaults to "English".
        """
        self.data_path = data_path
        self.preferred_lang = preferred_lang

    def __call__(self, note_list: List[Note], basic_info: Dict) -> Any:
        """Process a list of notes with basic user information.
        
        This method coordinates the overall data processing workflow, including
        splitting notes by type, refining data, converting to text, and extracting
        entities and relationships.
        
        Args:
            note_list: List of Note objects to process.
            basic_info: Dictionary containing basic user information.
            
        Returns:
            Processing results if any, otherwise None.
        """
        user_info, subjective_memory_notes, objective_memory_notes = self.split_notes_by_type(
            note_list, basic_info
        )

        subjective_notes_remade = self.refine_notes_data_subjective(
            subjective_memory_notes, user_info, self.data_path + "/L1/processed_data/subjective/note_remade.json"
        )

        objective_notes_remade = self.refine_notes_data_objective(
            objective_memory_notes, user_info, self.data_path + "/L1/processed_data/objective/note_remade.json"
        )

        self.json_to_txt_each(
            subjective_notes_remade,
            self.data_path + "/L1/processed_data/subjective",
            file_type="note",
        )

        self.json_to_txt_each(
            objective_notes_remade,
            self.data_path + "/L1/processed_data/objective",
            file_type="note",
        )

        logger.info("Data refinement completed, preparing to extract entities and relations")

        lang = user_info.get("lang", "English")

        if len(subjective_notes_remade) > 0:
            self.graphrag_indexing(
                subjective_notes_remade,
                self.data_path + "/L1/processed_data/subjective",
                self.data_path + "/L1/graphrag_indexing_output/subjective",
                lang,
            )

        if len(objective_notes_remade) > 0:
            self.graphrag_indexing(
                objective_notes_remade,
                self.data_path + "/L1/processed_data/objective",
                self.data_path + "/L1/graphrag_indexing_output/objective",
                lang,
            )
        return

    def gen_subjective_data(
            self,
            note_list: List[Note],
            data_output_base_dir: str,
            preference_output_path: str,
            diversity_output_path: str,
            selfqa_output_path: str,
            global_bio: str,
            topics_path: str,
            entitys_path: str,
            graph_path: str,
            user_name: str,
            config_path: str,
            user_intro: str,
            do_context: bool = False,
    ):
        """Generate subjective data for training.
        
        This method processes various types of subjective data including preferences,
        diversity, self-Q&A, and context data, then merges them into a single output file.
        
        Args:
            note_list: List of Note objects.
            data_output_base_dir: Base directory for output data.
            preference_output_path: Path to save preference data.
            diversity_output_path: Path to save diversity data.
            selfqa_output_path: Path to save self-Q&A data.
            global_bio: User's global biography.
            topics_path: Path to topics data.
            entitys_path: Path to entities data.
            graph_path: Path to graph data.
            user_name: Name of the user.
            config_path: Path to configuration file.
            user_intro: User's introduction.
        """
        preference_output_path = os.path.join(
            data_output_base_dir, preference_output_path
        )
        diversity_output_path = os.path.join(
            data_output_base_dir, diversity_output_path
        )
        selfqa_output_path = os.path.join(data_output_base_dir, selfqa_output_path)
        context_output_path = os.path.join(data_output_base_dir, "context_merged.json")

        logger.info("---" * 30 + "\nPreference data generating\n" + "---" * 30)
        self._gen_preference_data(topics_path, preference_output_path, global_bio)
        logger.info("---" * 30 + "\nPreference data generated\n" + "---" * 30)

        logger.info("---" * 30 + "\nDiversity data generating\n" + "---" * 30)
        self._gen_diversity_data(
            entitys_path,
            note_list,
            graph_path,
            diversity_output_path,
            user_name,
            global_bio,
            config_path,
        )
        logger.info("---" * 30 + "\nDiversity data generated\n" + "---" * 30)

        logger.info("---" * 30 + "\nSelfQA data generating\n" + "---" * 30)
        self._gen_selfqa_data(selfqa_output_path, user_name, user_intro, global_bio)
        logger.info("---" * 30 + "\nSelfQA data generated\n" + "---" * 30)

        if do_context:
            logger.info("---" * 30 + "\nContext data generating\n" + "---" * 30)
            self._gen_context_data(
                note_list,
                entitys_path,
                data_output_base_dir,
                user_name,
                user_intro,
                global_bio
            )
            logger.info("---" * 30 + "\nContext data generated\n" + "---" * 30)
            self._merge_context_data(data_output_base_dir, "context_merged.json")

        # Merge the four specified JSON files
        merged_data = []
        if do_context:
            json_files_to_merge = [
                preference_output_path,
                diversity_output_path,
                selfqa_output_path,
                context_output_path
            ]
        else:
            json_files_to_merge = [
                preference_output_path,
                diversity_output_path,
                selfqa_output_path,
            ]

        logger.info("---" * 30 + "\nMerging JSON files\n" + "---" * 30)

        for file_path in json_files_to_merge:
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            merged_data.extend(file_data)
                            logger.info(f"Added {len(file_data)} items from {file_path}")
                        else:
                            merged_data.append(file_data)
                            logger.info(f"Added 1 item from {file_path}")
                except Exception as e:
                    logger.error(f"Error merging file {file_path}: {str(e)}")
            else:
                if file_path == context_output_path and do_context == False:
                    continue
                logger.warning(f"File not found or path is None: {file_path}")

        # Save the merged data
        merged_output_path = os.path.join(data_output_base_dir, "merged.json")
        with open(merged_output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Merged data saved to {merged_output_path} with {len(merged_data)} total items")
        logger.info("---" * 30 + "\nJSON files merged\n" + "---" * 30)

    def _merge_context_data(self, data_output_base_dir: str, context_merged_file_name: str):
        """Merge context_enhanced.json and context_final.jsonl files.
        
        Args:
            data_output_base_dir: Base directory containing the files to merge.
            context_merged_file_name: Name of the output merged file.
        """
        logger.info("---" * 30 + "\nMerging context files\n" + "---" * 30)

        result = []

        # Process context_enhanced.json
        try:
            with open(f"{data_output_base_dir}/context_enhanced.json", 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)

            for item in enhanced_data:
                try:
                    # Try to parse context_enhanced_need field
                    enhanced_need = json.loads(item["context_enhanced_need"])
                    result.append({
                        "user_request": item["initial_need"],
                        "enhanced_request": enhanced_need["enhanced_request"]
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to process enhanced item: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error processing context_enhanced.json: {str(e)}")

        # Process context_final.jsonl
        try:
            with open(f"{data_output_base_dir}/context_final.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        final_item = json.loads(line)
                        try:
                            # Try to parse response field
                            response = json.loads(final_item["response"])
                            result.append({
                                "user_request": final_item["initial_need"],
                                "expert_response": final_item["expert_response"],
                                "user_feedback": response.get("feedback", "") if isinstance(response['feedback'],
                                                                                            str) else
                                response['feedback'][0]
                            })
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to process final item response: {str(e)}")
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line as JSON: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error processing context_final.jsonl: {str(e)}")

        # Save the merged results
        output_path = f"{data_output_base_dir}/{context_merged_file_name}"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"Merged context files saved to {output_path}")
        logger.info("---" * 30 + "\nContext files merged\n" + "---" * 30)

    def split_notes_by_type(self, note_list: List[Note], basic_info: Dict):
        """Split notes into subjective and objective categories.
        
        Args:
            note_list: List of Note objects to split.
            basic_info: Dictionary containing basic user information.
            
        Returns:
            Tuple of (user_info, subjective_notes, objective_notes).
        """
        user_info = {
            "username": basic_info["username"],
            "aboutMe": basic_info["aboutMe"],
            "statusBio": basic_info["statusBio"],
            "globalBio": basic_info["globalBio"],
            "lang": basic_info.get("lang", "English"),
        }

        subjective_notes = []
        objective_notes = []
        for note in note_list:
            if note.memory_type in SUBJECT_NOTE_TYPE:
                subjective_notes.append(note)
            elif note.memory_type in OBJECT_NOTE_TYPE:
                objective_notes.append(note)
            else:
                logger.warning(f"Note type not supported: {note.memory_type}")
                continue
        logger.info(
            f"Subjective notes: {len(subjective_notes)}, Objective notes: {len(objective_notes)}"
        )
        return user_info, subjective_notes, objective_notes

    def refine_notes_data_subjective(
            self, note_list: List[Note], user_info: Dict, json_file_remade: str
    ):
        """Refine subjective notes data and save to JSON.
        
        Args:
            note_list: List of Note objects to refine.
            user_info: Dictionary containing user information.
            json_file_remade: Path to save the refined JSON data.
            
        Returns:
            List of refined Note objects.
        """
        data_filtered = []

        lang = user_info.get("lang", "English")

        if lang not in SUBJECTIVE_TEMPLATES:
            lang = "English"

        selected_templates = SUBJECTIVE_TEMPLATES[lang]

        for note in note_list:
            if note.memory_type not in OBJECT_NOTE_TYPE:
                note.create_time = format_timestr(note.create_time)

                basic_template = random.choice(selected_templates["basic"]).format(user_name=user_info["username"])

                if note.insight:
                    # for markdown and doc
                    note.processed = basic_template + str(note.insight)
                else:
                    # for short text
                    note.processed = basic_template + str(note.content)

                if note.title:
                    note.processed += selected_templates["title_suffix"].format(title=note.title)

                data_filtered.append(note)

        logger.info(f"Refined subjective notes: {len(data_filtered)}")

        json_data_filted = [o.to_json() for o in data_filtered]
        file_dir = os.path.dirname(json_file_remade)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(json_file_remade, "w", encoding="utf-8") as file:
            json.dump(json_data_filted, file, ensure_ascii=False, indent=4)

        return data_filtered

    def refine_notes_data_objective(self, note_list: List[Note], user_info: Dict, json_file_remade: str):
        """Refine objective notes data and save to JSON.
        
        Args:
            note_list: List of Note objects to refine.
            user_info: Dictionary containing user information.
            json_file_remade: Path to save the refined JSON data.
            
        Returns:
            List of refined Note objects.
        """
        lang = user_info.get("lang", "English")
        if lang not in OBJECTIVE_TEMPLATES:
            lang = "English"

        templates = OBJECTIVE_TEMPLATES[lang]

        new_item_list = []
        for note in note_list:
            if note.memory_type in OBJECT_NOTE_TYPE:
                if note.insight is None or note.insight == "":
                    continue
                new_item = note.copy()
                if note.content is not None and note.content != "":
                    new_item.processed = random.choice(
                        templates["with_content"]
                    ).format(content=note.content, insight=note.insight)
                else:
                    new_item.processed = random.choice(
                        templates["without_content"]
                    ).format(insight=note.insight)
                new_item_list.append(new_item)

        logger.info(f"Refined objective notes: {len(new_item_list)}")

        file_dir = os.path.dirname(json_file_remade)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        with open(json_file_remade, "w", encoding="utf-8") as f:
            json.dump(new_item_list, f, ensure_ascii=False, indent=4)

        return new_item_list

    def json_to_txt_each(
            self, list_processed_notes: List[Note], txt_file_base: str, file_type: str
    ):
        """Convert processed notes from JSON to individual text files.
        
        Args:
            list_processed_notes: List of processed Note objects.
            txt_file_base: Base directory to save text files.
            file_type: Type of note for naming the output files.
        """
        # Ensure the target directory exists
        if not os.path.exists(txt_file_base):
            os.makedirs(txt_file_base)
            logger.warning("Currently running in function json_to_txt_each")
            logger.warning(f"Specified directory does not exist, created: {txt_file_base}")

        # Clear all existing files in the target directory
        for existing_file in os.listdir(txt_file_base):
            file_path = os.path.join(txt_file_base, existing_file)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Removed existing file: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {str(e)}")

        logger.info(f"Cleared all existing files in {txt_file_base}")

        for no, item in enumerate(tqdm(list_processed_notes)):
            # Build the txt file path
            txt_file = os.path.join(txt_file_base, f"{file_type}_{no}.txt")
            try:
                # Ensure the processed field exists in the item
                if item.processed:
                    with open(txt_file, "w", encoding="utf-8") as tf:
                        tf.write(item.processed)
                else:
                    logger.warning(f"Warning: 'processed' key missing for item {no}")

            except Exception as e:
                logger.error(traceback.format_exc())

    def graphrag_indexing(
            self, note_list: List[Note], graph_input_dir: str, output_dir: str, lang: str
    ):
        """Index notes using GraphRAG.
        
        This method configures and runs GraphRAG indexing on the processed notes,
        creating entity and relation extractions.
        
        Args:
            note_list: List of Note objects to index.
            graph_input_dir: Directory containing input files for indexing.
            output_dir: Directory to save indexing results.
            lang: Language for the prompts.
        """
        GRAPH_CONFIG = os.path.join(
            os.getcwd(), "lpm_kernel/L2/data_pipeline/graphrag_indexing/settings.yaml"
        )

        ENV_CONFIG = os.path.join(
            os.getcwd(), "lpm_kernel/L2/data_pipeline/graphrag_indexing/.env"
        )

        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()

        chat_api_key = user_llm_config.chat_api_key
        chat_base_url = user_llm_config.chat_endpoint
        chat_model_name = user_llm_config.chat_model_name

        embedding_api_key = user_llm_config.embedding_api_key
        embedding_base_url = user_llm_config.embedding_endpoint
        embedding_model_name = user_llm_config.embedding_model_name

        with open(GRAPH_CONFIG, "r", encoding="utf-8") as file:
            settings = yaml.safe_load(file)

        with open(ENV_CONFIG, "w", encoding="utf-8") as file:
            file.write(f"GRAPHRAG_API_KEY={chat_api_key}")

        settings["input"]["base_dir"] = graph_input_dir
        settings["output"]["base_dir"] = output_dir
        settings["reporting"]["base_dir"] = os.path.join(output_dir, "../report")

        settings["models"]["default_chat_model"]["api_base"] = chat_base_url
        settings["models"]["default_chat_model"]["model"] = chat_model_name
        settings["models"]["default_chat_model"]["api_key"] = chat_api_key

        if chat_model_name.startswith("openai"):
            settings["models"]["default_chat_model"]["model"] = chat_model_name.replace("openai/", "")

        if embedding_model_name.startswith("openai"):
            settings["models"]["default_embedding_model"]["model"] = embedding_model_name.replace("openai/", "")
        else:
            settings["models"]["default_embedding_model"]["model"] = embedding_model_name

        settings["models"]["default_embedding_model"]["api_base"] = embedding_base_url
        settings["models"]["default_embedding_model"]["api_key"] = embedding_api_key

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.warning(f"Specified output directory does not exist, created: {output_dir}.")

        with open(GRAPH_CONFIG, "w", encoding="utf-8") as file:
            yaml.dump(settings, file, default_flow_style=False, allow_unicode=True)

        logger.info(f"Input base_dir has been updated to {graph_input_dir} and saved.")
        logger.info(f"Output base_dir has been updated to {output_dir} and saved.")
        logger.info(
            f"Report base_dir has been updated to {os.path.join(output_dir, 'report')} and saved."
        )

        # Read prompts configuration and modify entity_extraction/summarize_descriptions from "in Chinese" to "in {lang}"
        entity_extraction_path = os.path.join(
            os.getcwd(),
            "lpm_kernel/L2/data_pipeline/graphrag_indexing/prompts/extract_graph.txt",
        )
        with open(entity_extraction_path, "r", encoding="utf-8") as f1:
            entity_extraction = f1.read()
            entity_extraction = entity_extraction.replace("<lang>", lang)
        with open(entity_extraction_path, "w", encoding="utf-8") as f2:
            f2.write(entity_extraction)

        summarize_descriptions_path = os.path.join(
            os.getcwd(),
            "lpm_kernel/L2/data_pipeline/graphrag_indexing/prompts/summarize_descriptions.txt",
        )
        with open(summarize_descriptions_path, "r", encoding="utf-8") as f1:
            summarize_descriptions = f1.read()
            summarize_descriptions = summarize_descriptions.replace("<lang>", lang)
        with open(summarize_descriptions_path, "w", encoding="utf-8") as f2:
            f2.write(summarize_descriptions)

        # Run GraphRAG indexing
        try:
            # graphrag index --config lpm_kernel/L2/data_pipeline/graphrag_indexing/settings.yaml --root lpm_kernel/L2/data_pipeline/graphrag_indexing
            
            logger.info(f"Running GraphRAG indexing... {os.environ['PYTHONEXE']}")
            result = subprocess.run(
                [
                    os.environ['PYTHONEXE'],
                    "-m",
                    "graphrag",
                    "index",
                    "--config",
                    os.path.join(
                        os.getcwd(),
                        "lpm_kernel/L2/data_pipeline/graphrag_indexing/settings.yaml",
                    ),
                    "--root",
                    os.path.join(
                        os.getcwd(),
                        "lpm_kernel/L2/data_pipeline/graphrag_indexing",
                    ),
                ],
                check=True,
                text=True,
                capture_output=True,
                shell=True,
            )
            logger.info(f"GraphRAG indexing completed successfully. {result}")
            logger.error(f"subprocess.run graphrag index error: {result.stderr}")
            if result.stderr:
                raise RuntimeError("subprocess.run graphrag index error")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

        """Post-processing"""

        self.creat_mapping(
            output_dir,
            note_list,
            os.path.join(
                os.getcwd(),
                "resources/L2/data_pipeline/raw_data/id_entity_mapping_subjective_v2.json",
            ),
        )

    def creat_mapping(self, graph_dir, note_list, mapped_json_file):
        """Create a mapping between entities and documents.
        
        Args:
            graph_dir: Directory containing GraphRAG output.
            note_list: List of Note objects.
            mapped_json_file: Path to save the mapping file.
        """
        try:
            document = pd.read_parquet(
                os.path.join(graph_dir, "documents.parquet")
            )
            entities = pd.read_parquet(
                os.path.join(graph_dir, "entities.parquet")
            )
        except Exception as e:
            return

        json_data = []

        # show the column names
        logger.info(f"Entity Column names: {entities.columns}")

        logger.info(f"Document Column names: {document.columns}")

        for e_i, e_r in tqdm(entities.iterrows(), total=len(entities)):
            json_item = {}
            json_item["entity_id"] = e_r["id"]
            json_item["entity_name"] = e_r["title"]
            json_item["entity_description"] = e_r["description"]
            json_item["doc_id"] = []
            text_unit_ids = e_r["text_unit_ids"]

            for text_unit_id in text_unit_ids:
                for d_i, d_r in document.iterrows():
                    if text_unit_id in d_r["text_unit_ids"]:
                        if "note" in d_r["title"]:
                            json_item["doc_id"].append(
                                note_list[
                                    int(
                                        d_r["title"]
                                        .replace(".txt", "")
                                        .replace("note_", "")
                                    )
                                ].id
                            )
            json_data.append(json_item)

        with open(os.path.join(mapped_json_file), "w", encoding="utf-8") as file:
            json.dump(json_data, file, ensure_ascii=False, indent=4)

    def _gen_preference_data(self, topics_path, preference_output_path, bio):
        """Generate preference data based on user topics and bio.
        
        Args:
            topics_path: Path to topics data file.
            preference_output_path: Path to save preference data.
            bio: User's bio information.
        """
        processor = PreferenceQAGenerator(
            filename=topics_path, bio=bio, preference_language=self.preferred_lang
        )
        processor.process_clusters(preference_output_path)

    def _gen_diversity_data(
            self,
            entitys_path,
            note_list: List[Note],
            graph_path,
            output_path,
            user_name,
            global_bio,
            config_path,
    ):
        """Generate diversity data based on entities and notes.
        
        Args:
            entitys_path: Path to entities data.
            note_list: List of Note objects.
            graph_path: Path to graph data.
            output_path: Path to save diversity data.
            user_name: Name of the user.
            global_bio: User's global biography.
            config_path: Path to configuration file.
        """
        processor = DiversityDataGenerator(self.preferred_lang)
        processor.generate_data(
            entitys_path, note_list, config_path, graph_path, user_name, global_bio, output_path
        )

    def _gen_selfqa_data(self, output_path, user_name, user_intro, bio):
        """Generate self Q&A data based on user information.
        
        Args:
            output_path: Path to save self Q&A data.
            user_name: Name of the user.
            user_intro: User's introduction.
            bio: User's bio information.
        """
        selfqa = SelfQA(
            user_name=user_name,
            user_input_introduction=user_intro,
            user_global_bio=bio,
            preferred_language=self.preferred_lang,
        )
        q_a_list = selfqa.generate_qa()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(q_a_list, f, ensure_ascii=False, indent=4)

    def _gen_context_data(
            self,
            note_list: List[Note],
            entitys_path: str,
            data_output_base_dir: str,
            user_name: str,
            user_intro: str,
            global_bio: str
    ):
        """Generate context-based conversation data.
        
        This method generates contextual conversation data in multiple steps:
        1. Generate initial context needs
        2. Enhance context data
        3. Generate expert responses
        4. Generate critic data
        
        Args:
            note_list: List of Note objects.
            entitys_path: Path to entities data.
            data_output_base_dir: Base directory for output data.
            user_name: Name of the user.
            user_intro: User's introduction.
            global_bio: User's global biography.
        """
        context_generator = ContextGenerator(
            preferred_language=self.preferred_lang,
            user_name=user_name,
            user_bio=global_bio
        )

        # 1. Generate initial context needs
        context_generator.generate_context_needs(
            note_list=note_list,
            entity_map_path=os.path.join(entitys_path),
            data_output_base_dir=data_output_base_dir,
            needs_file_name="context_needs.json"
        )
        logger.info("---" * 30 + "\nContext needs generated\n" + "---" * 30)
        logger.info(data_output_base_dir + "/context_needs.json")

        # 2. Generate context enhanced data
        context_generator.generate_context_enhance_data(
            data_output_base_dir=data_output_base_dir,
            needs_file_name="context_needs.json",
            context_enhanced_res_file_name="context_enhanced.json",
            note_list=note_list
        )
        logger.info("---" * 30 + "\nContext enhanced generated\n" + "---" * 30)
        logger.info(data_output_base_dir + "/context_enhanced.json")

        # 3. Generate expert responses
        context_generator.expert_response_generator(
            data_output_base_dir=data_output_base_dir,
            context_enhanced_res_file_name="context_enhanced.json",
            output_file_name="expert_responses.json"
        )
        logger.info("---" * 30 + "\nExpert responses generated\n" + "---" * 30)
        logger.info(data_output_base_dir + "/expert_responses.json")

        # 4. Generate expert responses and critic data
        context_generator.gen_context_critic_data(
            data_output_base_dir=data_output_base_dir,
            expert_response_file_name="expert_responses.json",
            out_file_name="context_final.jsonl"
        )
        logger.info("---" * 30 + "\nContext critic generated\n" + "---" * 30)
        logger.info(data_output_base_dir + "/context_final.jsonl")
