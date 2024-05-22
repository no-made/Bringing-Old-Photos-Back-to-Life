import errno

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import uuid
from django.conf import settings
import os.path
import sys
from PIL import Image
import torch
import shutil
import numpy as np
from subprocess import call
from Global.perform_inference import perform
from Global.detection_global import detect
from Face_Detection.detect_faces_dlib import detecting_faces
from Face_Detection.align_warp_back_multiple_dlib_HR_faces import align_warp_back_multiple
from Face_Enhancement.face_enhancement import enhance_face
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from django.views.decorators.csrf import csrf_exempt
import logging
import gc
media_folder = './static/media'
input_folder = './static/media/input_images'
input_scratched = './static/media/input_scratched_images'
input_hd = './static/media/input_hd_images'

output_folder = './static/media/output_images'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"
logger = logging.getLogger(__name__)

checkpoint_name = "Setting_9_epoch_100"  # "Setting_9_epoch_100"

def index(response):
    return HttpResponse("This message is from Restoring App")


def landing(request):
    print('The landing page has been called.')
    user_id = str(uuid.uuid4())[:8]  # Generate a unique user ID
    protocol = settings.PROTOCOL
    host = settings.HOST
    port = settings.PORT
    GA_MEASUREMENT_ID = settings.GA_MEASUREMENT_ID
    print(f'GA_MEASUREMENT_ID_VIEWS: {GA_MEASUREMENT_ID}')
    request.session['user_id'] = user_id  # Store the user ID in the session
    return render(request, 'landing.html', {'user_id': user_id, 'protocol': protocol, 'host': host, 'port': port, 'GA_MEASUREMENT_ID': GA_MEASUREMENT_ID})

@api_view(['DELETE'])
@authentication_classes([])
@permission_classes([])
def delete_temp_folder(request):
    print('The DELETE temp folder has been called.')
    user_id = request.headers.get('X-USER-ID')
    folder_path = f'./static/media/{user_id}'
    shutil.rmtree(folder_path, ignore_errors=True)
    print(f'Temp folder for user {user_id} has been deleted.')
    return JsonResponse({'message': f'Temp folder for user {user_id} has been deleted.'})

@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def upload_image(request):
    print('The upload image has been called.')
    global input_folder, input_scratched, input_hd, media_folder, output_folder
    user_id = request.headers.get('X-USER-ID')
    media_folder = './static/media/' + user_id
    input_folder = media_folder + '/input_images'
    input_scratched = media_folder + '/input_scratched_images'
    input_hd = media_folder + '/input_hd_images'
    output_folder = media_folder + '/output_images'
    folders = [media_folder, input_folder, input_scratched, input_hd, output_folder]
    processed_images = []
    for folder in folders:
        delete_and_make_folder(folder)
    print(request.FILES)
    files = request.FILES.getlist('base')
    scratched = request.POST.getlist('scratched')
    hd_files = request.POST.getlist('hd')

    if not files:
        return JsonResponse({'error': 'No images found'})

    for i in range(len(files)):
        image = files[i]
        print(type(str(image)), str(image))
        if image.content_type not in ['image/jpeg', 'image/png']:
            return JsonResponse({'error': 'Invalid image format'})
        img = Image.open(image)
        print('image', img.size, img.mode,)
        if scratched[i] == 'true' and hd_files[i] == 'true':
            input_filename = os.path.join(input_hd, str(image))
        elif scratched[i] == 'true' and hd_files[i] == 'false':
            input_filename = os.path.join(input_scratched, str(image))
        else:
            input_filename = os.path.join(input_folder, str(image))
        os.makedirs(os.path.dirname(input_filename), exist_ok=True)
        img.save(input_filename)

    modify()
    for file in os.listdir(os.path.join(output_folder, 'stage_1_restore_output', 'input_image')):
        processed_images.append(file)
    differences(input_folder)
    differences(input_scratched)
    differences(input_hd)
    return JsonResponse({'images': processed_images})


def modify(image_filename=None, cv2_frame=None, scratched=None):
    # gpu = get_gpu()
    gpu = -1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    main_environment = os.getcwd()
    # Stage 1: Overall Quality Improve
    stage_1_processing(main_environment, gpu)

    # Stage 2: Face Detection
    stage_2_processing(output_folder)

    # Stage 3: Face Restore
    stage_3_processing(main_environment, output_folder, gpu)

    # Stage 4: Warp back
    stage_4_processing(main_environment, output_folder)

    print("All the processing is done. Please check the results.")
    os.chdir(".././")


def run_cmd(command):
    torch.cuda.empty_cache()
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


def get_gpu():
    gpu = -1
    max_gpu_memory = 0

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print('Number of GPUs available:', num_gpus)

        for i in range(num_gpus):
            cuda_properties = torch.cuda.get_device_properties(i)
            gpu_memory = cuda_properties.total_memory / 1024 ** 3  # Convert bytes to gigabytes

            if gpu_memory > 4 and gpu_memory > max_gpu_memory:
                gpu = i
                max_gpu_memory = gpu_memory

        if gpu == -1:
            print('No suitable GPU with more than 4 GB of RAM found. Setting gpu = -1')
        else:
            print('Selected GPU:', gpu, torch.cuda.get_device_name(gpu), torch.cuda.mem_get_info(gpu))
    else:
        print('No GPU available. Setting gpu = -1')

    return gpu


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def stage_1_processing(main_env, gpu):
    # Stage 1: Overall Quality Improve
    # logger.debug('Running Stage 1.')
    print("Running Stage 1: Overall restoration", main_env)
    os.chdir(os.path.join(main_env, "Global"))
    print("current directory: ", os.getcwd())

    stage_1_input_dir = os.path.join("..", input_folder)
    stage_1_scratched_dir = os.path.join("..", input_scratched)
    stage_1_hd_dir = os.path.join("..", input_hd)
    stage_1_output_dir = os.path.join("..", output_folder, "stage_1_restore_output")
    print("stage 1 output folder: ", stage_1_output_dir)

    create_directory_if_not_exists(stage_1_output_dir)
    # Process scratched images
    if not len(stage_1_scratched_dir)==0:
        print('precessing scratches')
        process_image(stage_1_scratched_dir, stage_1_output_dir, gpu, False)

    # Process HD images
    if not len(stage_1_hd_dir) == 0:
        process_image(stage_1_hd_dir, stage_1_output_dir, gpu, True)

    # Process regular input images
    if not len(stage_1_input_dir) == 0:
        process_image(stage_1_input_dir, stage_1_output_dir, gpu, False)

    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join("..", output_folder, "final_output")
    create_directory_if_not_exists(stage_4_output_dir)

    # Copy results to the final output folder
    copy_results_to_final_output(stage_1_results, stage_4_output_dir)

    print("Finish Stage 1 ...")
    print("\n")


def stage_2_processing(out_folder):
    # Stage 2: Face Detection
    print("Running Stage 2: Face Detection")
    os.chdir(".././Face_Detection")
    print("stage 2 current dir: ", os.getcwd())
    stage_2_input_dir = os.path.join("..", out_folder, "stage_1_restore_output", "restored_image")
    stage_2_output_dir = os.path.join("..", out_folder, "stage_2_detection_output")
    print("stage 2 input folder: ", stage_2_input_dir)
    print("stage 2 output folder: ", stage_2_output_dir)

    create_directory_if_not_exists(stage_2_output_dir)
    detecting_faces(stage_2_input_dir, stage_2_output_dir)
    print("Finish Stage 2 ...")
    print("\n")


def stage_3_processing(main_environment, out_folder, gpu):
    # Stage 3: Face Restore
    print("Running Stage 3: Face Enhancement")
    os.chdir(os.path.join(main_environment, "Face_Enhancement"))
    print("stage 3 current dir: ", os.getcwd())

    stage_3_input_mask = "./"
    stage_3_input_face = os.path.join("..", out_folder, "stage_2_detection_output")
    stage_3_output_dir = os.path.join("..", out_folder, "stage_3_face_output")
    print("stage 3 input face folder: ", stage_3_input_face)
    print("stage 3 input mask folder: ", stage_3_input_mask)
    print("stage 3 output folder: ", stage_3_output_dir)

    create_directory_if_not_exists(stage_3_output_dir)
    enhance_face(stage_3_output_dir, stage_3_input_face, stage_3_input_mask, checkpoint_name, gpu)

    print("Finish Stage 3 ...")
    print("\n")


def stage_4_processing(main_environment, out_folder):
    # Stage 4: Warp back
    logger.debug('Running Stage 4.')
    print("Running Stage 4: Blending")
    os.chdir(os.path.join(main_environment, "Face_Detection"))
    print("stage 4 current dir: ", os.getcwd())
    stage_4_input_image_dir = os.path.join("..", out_folder, "stage_1_restore_output", "restored_image")
    stage_4_input_face_dir = os.path.join("..", out_folder, "stage_3_face_output", "each_img")
    stage_4_output_dir = os.path.join("..", out_folder, "final_output")

    print("stage 4 input image folder: ", stage_4_input_image_dir)
    print("stage 4 input face folder: ", stage_4_input_face_dir)
    print("stage 4 output folder: ", stage_4_output_dir)

    create_directory_if_not_exists(stage_4_output_dir)
    # align_warp_back_multiple(stage_4_input_image_dir, stage_4_input_face_dir, stage_4_output_dir)
    stage_4_command = (
        f"python align_warp_back_multiple_dlib.py --origin_url {stage_4_input_image_dir} "
        f"--replace_url {stage_4_input_face_dir} --save_url {stage_4_output_dir}"
    )

    run_cmd(stage_4_command)
    print("Finish Stage 4 ...")
    print("\n")


def copy_results_to_final_output(stage_1_results, stage_4_output_dir):
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)


def process_image(input_dir, output_dir, gpu, is_hd=False):
    if 'input_images' in input_dir and not len(os.listdir(input_dir)) == 0:
        print('QualityReostoring...')
        perform(input_dir, "", "Full", output_dir, Quality_restore=True, Scratch_and_Quality_restore=False, HR=is_hd, gpu_ids=gpu)
        # stage_1_command = (
        #         "python test.py --test_mode Full --Quality_restore --test_input "
        #         + input_dir
        #         + " --outputs_dir "
        #         + output_dir
        #         + " --gpu_ids "
        #         + gpu
        # )
        # run_cmd(stage_1_command)
    elif not len(os.listdir(input_dir)) == 0:
        mask_dir = os.path.join(output_dir, "masks_hd") if is_hd else os.path.join(output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        detect(input_dir, mask_dir, 'full_size', gpu)
        perform(new_input, new_mask, "", output_dir, Quality_restore=False, Scratch_and_Quality_restore=True, HR=is_hd, gpu_ids=gpu)


def delete_and_make_folder(folder):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
        except OSError as e:
            if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                print(f"Error: {folder} : {e.strerror}")
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:  # errno.EEXIST = file already exists
            print(f"Error: {folder} : {e.strerror}")


def differences(folder):
    check_folder = os.path.join(output_folder, 'stage_1_restore_output', 'input_image')
    print("current dir: ", os.getcwd())
    print("differences stage - check folder: ", check_folder)
    check_files = os.listdir(check_folder)
    for file in os.listdir(folder):
        name, ext = os.path.splitext(file)
        found = False
        for filename in check_files:
            if os.path.splitext(filename)[0] == name:
                ext = os.path.splitext(filename)[1]
                found = True
                break

        if not found:
            print(f"Skipping {file} - corresponding file not found in check folder")
            continue
        input_src = os.path.join(folder, file)
        output_src = os.path.join(output_folder, 'final_output', name + ext)
        input_f = os.path.join(media_folder, name + '_input' + ext)
        output_f = os.path.join(media_folder, name + '_output' + ext)
        paragon_f = os.path.join(media_folder, name + '_paragon' + ext)

        shutil.copy(input_src, input_f)
        shutil.copy(output_src, output_f)

        input_image = Image.open(input_f)
        output_image = Image.open(output_f)

        if input_image.size != output_image.size:
            output_image = output_image.resize(input_image.size)
            output_image.save(output_f)

        input_gray = np.dot(np.array(input_image)[..., :3], [0.2989, 0.5870, 0.1140])
        output_gray = np.dot(np.array(output_image)[..., :3], [0.2989, 0.5870, 0.1140])

        result = np.maximum(input_gray - output_gray, 0)

        result_image = Image.fromarray(result.astype('uint8'))
        result_image.save(paragon_f)
