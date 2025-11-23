for i in range(10):
    prompt_from_user = input("Enter a prompt: find similar cell in the sample ")
    ai_agent_interpretation = ai_agent_interpret(prompt_from_user)
    find_siimlar_cell_result = microscope_control_script("move the stage, scan the sample, segment the sample, find similar cell in the sample")
    result_interpolation = ai_agent_interpolation(find_siimlar_cell_result)
    

Goal: find 400 similar cells
For: 
1. move stage and scan the sample
2. segment, generate embedding, upload to weaviate, similarity search, get the result
3. check the similarity result, decide current similarity search result is similar or not, abandon unsimilar cells
4. if the similarity result is not enough, repeat the process

from typing import Union, Optional, Tuple

def find_similar_cells(
    image,
    text_description_of_the_cell: Optional[str] = None,
    cell_uuid_and_app_id: Optional[Tuple[str, str]] = None
):
    
    similar_cells = []
    stop = False
    for well_id in range(A1, E12):
        microscope_move_to_well(well_id)
        for x,y in range(scan_region_in_the_well):
            microscope_move_stage(x,y)
            image = microscope_acquire_image()
            cell_images = segment_cells_from_image(image)

            embedding_vectors = generate_embeddings_from_cells(cell_images)
            for embedding_vector in embedding_vectors:
                similarity_score = compare_similairty_with_target_cell(embedding_vector)
                if similarity_score > threshold:
                    similar_cells.append(embedding_vector, similarity_score)
                    if len(similar_cells) < 400:
                        continue
                    else:
                        stop = True
                        break
                    
            if len(similar_cells) < 400:
                continue
            else:
                stop = True
                break 
        if stop:
            break
