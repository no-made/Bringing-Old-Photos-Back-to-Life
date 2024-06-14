let choice = "gallery"//"gallery"; // or "input"
let selected_images = [];
let scratched_images = [];
let hd_images = [];
let fileList = [];


let mainSection, user_id, protocol, hostAddress, port, host, fileNames;
$(document).ready(function () {
    ({user_id, port, host, protocol, location: {hostname: hostAddress}} = window);
    console.log('protocol:', protocol, 'host:', host, 'port:', port, 'user_id:', user_id);

    mainSection = $("#central");
    mainSection.html(getLandingSection());

    $("#start_button").click(onStartButtonClick);

    window.addEventListener('beforeunload', async (event) => {
        event.preventDefault();
        await deleteTempFolder(user_id, protocol, hostAddress);
    });
});

function getLandingSection() {
    return `
        <div id='landing'>
            <h1>Welcome to ReInHerit's Old Photos' Restorer!</h1>
            <p>Our virtual toolkit has been designed to assist museum owners and enthusiasts in effortlessly restoring old photos.</p>
            <p>By simply uploading your photo, whether it has scratches or not, our advanced processing algorithms will work their magic.
                Once the image processing is complete, you will have a fully restored photo to cherish and share with future generations.<br>
                Try it out today and rediscover the beauty of your old photographs!
            </p>
            <img id='cover_image' alt='landing image' src='static/assets/images/scratch_detection.png'>
            <a id='start_button' class='square_btn'>START TO RESTORE</a>
        </div>`;
}

function getInputSection() {
    return `
        <div id='input-section'>
            <div id='input-description'>
                <h1>INPUT</h1>
                <p class='vertical'>To load images from your hard disk, browse to the folder where they're stored and select the ones you want.
                    You can load as many images as you'd like from the same folder. However, keep in mind that the more files you choose
                    and the larger their size, the longer it will take to process and load them. So, please be patient while the files are being processed.
                    <br>
                    If a photo has scratches or damage that needs to be repaired, select the 'with scratches' checkbox.
                    And if the image with scratches has a DPI (dots per inch) of 300 or higher, select the checkbox labeled 'is HD'.
                </p>
            </div>
            <form id='image-form'>
                <a class='square_btn' type='button' onclick="document.getElementById('image-input').click(); return false;"> BROWSE </a>
                <input type='file' id='image-input' name='image' class='hidden_tag' accept='image/*' multiple>
                <div id='selected-images'></div>  
                <br>              
                <a id='submit_label' class='square_btn hidden_tag' type='button' onclick="document.getElementById('submit-button').click(); return false;"> PROCESS </a>
                <button id='submit-button' type='submit' class='hidden_tag'></button>
                <br>
            </form>
            <div id='enlarged' class='hidden_tag'></div>
        </div>`;
}

function getLoadingSection() {
    return `
        <div id='loader'>
            <p id='shadow'></p>
            <div id='shadow'></div>
            <div id='box'></div>
            <input type='text' id='loading-text' value='Starting process...' />
        </div>`;
}

function getOutputSection() {
    return `
        <div id='output-section'>
            <div id='output-description'>
                <h1>OUTPUT</h1>
                <div class='vertical'>
                    <p>Here you can see the results of the processing:</p>
                    <ul style='list-style-type:disc;'>
                        <li>The first image is the original image</li>
                        <li>The second is an image of comparison on the areas most affected by the process.</li>
                        <li>The third image is the output image</li>
                    </ul>
                    <p>If you click the DOWNLOAD button, the app saves the outputs to your PC and returns to the home page.</p>
                    <p>If you click the RESTART button, the app returns to the home page and deletes all the processed images.</p>
                </div>
            </div>
            <div id='output-images'></div>
            <div id='buttons'>
                <a id='download_button' class='square_btn'>DOWNLOAD</a>
                <p id='or'> or </p>
                <a id='restart_button' class='square_btn'>RESTART</a>
            </div>
            <div id='enlarged' class='hidden_tag'> </div>
        </div>
    </div>`;
}

function getGallerySection() {
    return `
        <div id='gallery-section'>
            <div id='gallery-description'>
                <h1>GALLERY</h1>
                <p class='vertical'>Select one or more images from the gallery below checking the 'Select for processing' checkbox.
                    If a photo has scratches or damage that needs to be repaired, select the 'with scratches' checkbox. 
                    Click the PROCESS button to start the restoration process.
                </p>
            </div>
            <div id="gallery">
                <div id='gallery-images'></div>
                <br>
                <a id='submit_label' class='square_btn' type='button' onclick="document.getElementById('submit-button').click(); return false;"> PROCESS </a>
                <button id='submit-button' type='submit' class='hidden_tag'></button>
                <br><br>
            </div>
            <div id='enlarged' class='hidden_tag'></div>
        </div>`;
}

function onStartButtonClick() {
    // Here you define the condition or get the user input to choose between gallery and input section
    let selectedSection;
    if (choice === "gallery") {
        selectedSection = getGallerySection();
    } else {
        selectedSection = getInputSection();
    }

    // const mainSection = $("#central");
    mainSection.html(selectedSection);

    if (choice === "gallery") {
        fetchGalleryImages();
    } else {
        initializeInputSection();
    }
}

function initializeInputSection() {
    let data = {
        files: {},
        fileList: []
    };

    $("#image-input").change(function(event) {
        onImageInputChange(event, data);
    });
    $("#image-form").submit(function (event) {
        onImageFormSubmit(event, data.fileList)
    });
}

function onImageInputChange(event, data) {
    const selectedFiles = event.target.files;
    for (let i = 0; i < selectedFiles.length; i++) {
        const fileType = selectedFiles[i].type;
        if (fileType !== 'image/jpeg' && fileType !== 'image/png') {
            alert('Only jpg and png images are allowed!');
            return;
        }
    }
    data.files = updateFilesObject(data.files, selectedFiles);

    const dataTransfer = new DataTransfer();

    $("#submit_label").removeClass('hidden_tag');

    for (let file of Object.values(data.files)) {
        dataTransfer.items.add(new File([file], file.name, {type: file.type, lastModified: file.lastModified}));
    }

    data.fileList = dataTransfer.files;

    processFiles(data.fileList, data.files);
}

function updateFilesObject(files, selectedFiles) {
    const newFiles = {...files};
    for (let i = 0; i < selectedFiles.length; i++) {
        if (!Object.values(newFiles).some(e =>
            e.name === selectedFiles[i].name &&
            e.size === selectedFiles[i].size &&
            e.type === selectedFiles[i].type)) {
            const newKey = Object.keys(newFiles).length;
            newFiles[newKey] = selectedFiles[i];
        }
    }
    return newFiles;
}

function onImageFormSubmit(event, fileList) {
    event.preventDefault();
    // console.log('files: ', files, 'fileList: ', fileList);
    const {formData, fileNames} = prepareFormData(fileList);
    // const formData = new FormData();
    //
    // fileNames = []
    //
    // for (let i = 0; i < fileList.length; i++) {
    //     const fileName = fileList[i].name;
    //     const scratched = scratched_images[i] !== '' ? 'true' : 'false';
    //     const hd = hd_images[i] !== '' ? 'true' : 'false';
    //     fileNames.push({'name': fileName, 'scratched': scratched, 'hd': hd});
    //     formData.append('base', fileList[i]);
    //     formData.append('scratched', scratched);
    //     formData.append('hd', hd);
    // }
    showLoadingSection();
    // const loading_div = $(getLoadingSection());
    // mainSection.html(loading_div);
    //
    // const loading_text = document.getElementById('loading-text');
    // const loading = $("#loader");
    // const messages = ['Loading...', 'Please wait...', 'Almost done...', 'Hang tight...', 'Wow, it is quite large!', 'Oh my goodness!', 'What a tremendous size it is!'];
    // let index = 0;
    // setInterval(() => {
    //     loading_text.value = messages[index];
    //     index = (index + 1) % messages.length;
    // }, 5000);

    // fetch(`${protocol}://${hostAddress}:8000/upload/image/`, {
    //     method: 'POST',
    //     headers: {
    //         'X-User-Id': user_id,
    //         'Access-Control-Allow-Origin': '*'
    //     },
    //     body: formData,
    //     timeout: 65000 // timeout in milliseconds
    // })
    //     .then(response => {
    //         if (response.ok) {
    //             console.log('Image uploaded successfully');
    //             return response.json();
    //         } else {
    //             console.error('Error uploading image');
    //         }
    //     })
    uploadImages(formData)
        .then(data => handleUploadSuccess(data, fileList, fileNames))
        // .then(data => {
        //     console.log(data)
        //     const output_section = getOutputSection();
        //     mainSection.html(output_section);
        //     document.getElementById('output-section').style.overflowX = 'hidden';
        //     document.getElementById('output-section').style.position = 'relative';
        //     const output_images = $("#output-images");
        //
        //     const dataImageNames = data["images"].map(fileName => {
        //         const file_name = fileName.split("\\").pop();
        //         const name = file_name.split(".")[0];
        //         console.log(name, file_name)
        //         return name;
        //     });
        //     console.log(dataImageNames)
        //     for (let i = 0; i < fileList.length; i++) {
        //         const fileName = fileNames[i].name;
        //         const is_scratch = fileNames[i].scratched;
        //         const is_hd = fileNames[i].hd;
        //         const type = is_hd === 'true' ? '_hd_' : is_scratch === 'true' ? '_scratched_' : '_';
        //         const input_extension = fileName.split(".")[1];
        //         const ext = 'png';
        //         const name = fileName.split(".")[0];
        //
        //         if (!dataImageNames.includes(name)) {
        //             const folder_name = 'input' + type + 'images/';
        //
        //             const container = document.createElement("div");
        //             container.classList.add("unavailable_container");
        //             container.style.position = "relative";
        //             container.style.display = "inline-block";
        //             container.style.height = "200px";
        //             container.style.margin = "10px";
        //
        //             const input_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${folder_name}${fileName}`, "100%", null);
        //             input_image.style.filter = "grayscale(100%)";
        //             input_image.style.height = "100%";
        //
        //             const textOverlay = document.createElement("div");
        //             textOverlay.innerHTML = "This server has not GPU with enough memory to process this image.";
        //             textOverlay.classList.add("unavailable_text");
        //             textOverlay.style.position = "absolute";
        //             textOverlay.style.bottom = "0";
        //             textOverlay.style.left = "0";
        //             textOverlay.style.width = "100%";
        //             textOverlay.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
        //             textOverlay.style.color = "white";
        //             textOverlay.style.padding = "10px";
        //
        //             container.append(input_image);
        //             container.append(textOverlay);
        //             output_images.append(container);
        //         } else {
        //             const output_strips = document.createElement("div");
        //             output_strips.classList.add("output_strips");
        //
        //             const fileBaseName = fileName.split(".")[0];
        //
        //             const input_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${fileBaseName}_input.png`, 200, enlargeImage);
        //             const output_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${fileBaseName}_output.png`, 200, enlargeImage);
        //             const paragon_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${fileBaseName}_paragon.png`, 200, enlargeImage);
        //
        //
        //             output_strips.append(input_image);
        //             output_strips.append(paragon_image);
        //             output_strips.append(output_image);
        //             output_images.append(output_strips);
        //         }
        //     }
        //
        //     const download_button = document.getElementById("download_button");
        //     const restart_button = document.getElementById("restart_button");
        //
        //     download_button.addEventListener("click", async function () {
        //         await downloadAllImages(user_id, protocol, host);
        //     });
        //
        //     restart_button.addEventListener("click", async function () {
        //         reloadPage();
        //     });
        //
        // })
        .catch(error => {
            console.error('Error uploading image', error);
        })

}
function prepareFormData(fileList) {
    const formData = new FormData();
    const fileNames = [];
    let scratched, hd;
    console.log(scratched_images, hd_images)
    for (let i = 0; i < fileList.length; i++) {
        const fileName = fileList[i].name;
        if (choice === 'input') {
            scratched = scratched_images[i] !== '' ? 'true' : 'false';
            hd = hd_images[i] !== '' ? 'true' : 'false';}
        else if (choice === 'gallery') {
            const scratchedIndex = scratched_images.indexOf(fileName);
            const hdIndex = hd_images.indexOf(fileName);
            scratched = scratchedIndex !== -1 ? 'true' : 'false';
            hd = hdIndex !== -1 ? 'true' : 'false';
        }
        fileNames.push({ 'name': fileName, 'scratched': scratched, 'hd': hd });
        formData.append('base', fileList[i]);
        formData.append('scratched', scratched);
        formData.append('hd', hd);
        console.log('file: ', fileNames[i], 'scratched: ', scratched, 'hd: ', hd)
    }

    return { formData, fileNames };
}
function showLoadingSection() {
    const loadingDiv = $(getLoadingSection());
    mainSection.html(loadingDiv);

    const loadingText = document.getElementById('loading-text');
    const messages = ['Loading...', 'Please wait...', 'Almost done...', 'Hang tight...', 'Wow, it is quite large!', 'Oh my goodness!', 'What a tremendous size it is!'];
    let index = 0;
    setInterval(() => {
        loadingText.value = messages[index];
        index = (index + 1) % messages.length;
    }, 5000);
}
function uploadImages(formData) {
    return fetch(`${protocol}://${hostAddress}:8000/upload/image/`, {
        method: 'POST',
        headers: {
            'X-User-Id': user_id,
            'Access-Control-Allow-Origin': '*'
        },
        body: formData,
        timeout: 65000 // timeout in milliseconds
    })
        .then(response => {
            if (response.ok) {
                console.log('Image uploaded successfully');
                return response.json();
            } else {
                console.error('Error uploading image');
            }
        });
}
function handleUploadSuccess(data, fileList, fileNames) {
    console.log(data);
    const outputSection = getOutputSection();
    mainSection.html(outputSection);
    document.getElementById('output-section').style.overflowX = 'hidden';
    document.getElementById('output-section').style.position = 'relative';
    const outputImages = $("#output-images");

    const dataImageNames = data["images"].map(fileName => {
        const file_name = fileName.split("\\").pop();
        const name = file_name.split(".")[0];
        console.log(name, file_name);
        return name;
    });

    console.log(dataImageNames);

    for (let i = 0; i < fileList.length; i++) {
        const name = fileNames[i].name;
        const scratched = fileNames[i].scratched;
        const hd = fileNames[i].hd;
        const type = hd === 'true' ? '_hd_' : scratched === 'true' ? '_scratched_' : '_';
        // const inputExtension = name.split(".")[1];
        // const ext = 'png';
        const baseName = name.split(".")[0];

        if (!dataImageNames.includes(baseName)) {
            displayUnavailableImage(outputImages, baseName, name, type);
        } else {
            displayProcessedImages(outputImages, baseName, name);
        }
    };

    setupButtons();
}

function displayUnavailableImage(outputImages, baseName, name, type) {
    const folderName = 'input' + type + 'images/';

    const container = document.createElement("div");
    container.classList.add("unavailable_container");
    container.style.position = "relative";
    container.style.display = "inline-block";
    container.style.height = "200px";
    container.style.margin = "10px";

    const inputImage = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${folderName}${name}`, "100%", null);
    inputImage.style.filter = "grayscale(100%)";
    inputImage.style.height = "100%";

    const textOverlay = document.createElement("div");
    textOverlay.innerHTML = "This server has not GPU with enough memory to process this image.";
    textOverlay.classList.add("unavailable_text");
    textOverlay.style.position = "absolute";
    textOverlay.style.bottom = "0";
    textOverlay.style.left = "0";
    textOverlay.style.width = "100%";
    textOverlay.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
    textOverlay.style.color = "white";
    textOverlay.style.padding = "10px";

    container.append(inputImage);
    container.append(textOverlay);
    outputImages.append(container);
}

function displayProcessedImages(outputImages, baseName, name) {
    const outputStrips = document.createElement("div");
    outputStrips.classList.add("output_strips");

    const inputImage = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${baseName}_input.png`, 200, enlargeImage);
    const outputImage = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${baseName}_output.png`, 200, enlargeImage);
    const paragonImage = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${baseName}_paragon.png`, 200, enlargeImage);

    outputStrips.append(inputImage);
    outputStrips.append(paragonImage);
    outputStrips.append(outputImage);
    outputImages.append(outputStrips);
}

function setupButtons() {
    const downloadButton = document.getElementById("download_button");
    const restartButton = document.getElementById("restart_button");

    downloadButton.addEventListener("click", async function () {
        await downloadAllImages(user_id, protocol, host);
    });

    restartButton.addEventListener("click", async function () {
        reloadPage();
    });
}
function reloadPage() {
    location.reload();
}

function processFiles(fileList, files) {
    for (let i = 0; i < fileList.length; i++) {
        console.log('file ', i, ': ', fileList[i].name)
        if (!selected_images.some(e =>
            e.name === fileList[i].name &&
            e.size === fileList[i].size &&
            e.type === fileList[i].type)) {
            selected_images.push(fileList[i])

            const reader = new FileReader();

            // add a load event listener to the file reader
            reader.onload = function (event) {
                const image_name = fileList[i].name;
                scratched_images.push('');
                hd_images.push('');

                const image = createImageElement(event.target.result, 200, enlargeImage)


                const image_div = document.createElement("div");
                image_div.id = `${image_name}_div`;

                const checkbox_1 = createCheckboxElement(`check_${image_name}`, `checkboxChanged('${image_name}', '${i}', 'scratched')`, 'with scratches');
                const checkbox_2 = createCheckboxElement(`check_hd_${image_name}`, `checkboxChanged('${image_name}', '${i}', 'hd')`, 'is HD');
                checkbox_2.querySelector('input').disabled = true;
                image_div.appendChild(image);
                image_div.appendChild(checkbox_1);
                image_div.appendChild(checkbox_2);

                $("#selected-images").append(image_div);

                // checkbox_1.querySelector('input').addEventListener('change', function() {
                //     checkboxChanged(image_name, i, 'scratched');
                // });
                // checkbox_2.querySelector('input').addEventListener('change', function() {
                //     checkboxChanged(image_name, i, 'hd');
                // });
            };
            // read the selected file as a data URL
            reader.readAsDataURL(files[i]);
        }
    }
}

function checkboxChanged(file_name, number, which) {
    // Check if 'scratched' checkbox is selected
    const scratchedCheckbox = document.getElementById(`check_${file_name}`);
    const scratchedChecked = scratchedCheckbox.checked;

    // Activate 'hd' checkbox if 'scratched' is selected
    const hdCheckbox = document.getElementById(`check_hd_${file_name}`);
    if (scratchedChecked) {
        hdCheckbox.disabled = false;
    } else {
        hdCheckbox.disabled = true;
        hdCheckbox.checked = false;
    }
    console.log(scratched_images.indexOf(file_name), scratched_images[number])
    if (which === 'scratched') {
        if (scratched_images.indexOf(file_name) !== -1) {
            scratched_images[number] = ''
            console.log('removed: ', scratched_images)
        } else {
            scratched_images[number] = file_name
            console.log('added scratched: ', scratched_images)
        }
    } else if (which === 'hd') {
        if (hd_images.indexOf(file_name) !== -1) {
            hd_images[number] = ''
            console.log('removed: ', hd_images)
        } else {
            hd_images[number] = file_name
            console.log('added: ', hd_images)
        }
    }
    console.log('in',scratched_images)
}

function fetchGalleryImages() {
    // const {protocol, location: {hostname: hostAddress}, user_id} = window;

    fetch(`${protocol}://${hostAddress}:8000/load/gallery/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-User-Id': user_id,
            'Access-Control-Allow-Origin': '*'
        }
    }).then(response => {
        if (response.ok) {
            console.log('Gallery images loaded successfully');
            return response.json();
        } else {
            console.error('Error loading gallery images');
        }
    }).then(data => {
        const galleryImages = $("#gallery-images");
        console.log(data);
        let counter = 0; // Add this line

        for (let i = 0; i < data.length; i++) {
            const innerArray = data[i];
            for (let j = 0; j < innerArray.length; j++) {
                const filePath = innerArray[j].replaceAll("\\", "/");;
                const fileName = filePath.split("/").pop();
                console.log(fileName, filePath)
                scratched_images.push('');
                hd_images.push('');
                const image = createImageElement(filePath, 200, enlargeImage);

                const imageDiv = document.createElement("div");
                imageDiv.id = `${fileName}_div`;

                const checkbox1 = createCheckboxElement(`check_${fileName}`, `checkboxChanged('${fileName}', '${counter}', 'scratched')`, "with scratches", "40px");
                const checkbox2 = createCheckboxElement(`check_hd_${fileName}`, `checkboxChanged('${fileName}', '${i}', 'hd')`, "is HD", "40px");
                const selectCheckbox = createCheckboxElement(`select_${fileName}`, `selectImage('${filePath}', '${fileName}')`, "Select for processing", "40px");

                imageDiv.append(image, checkbox1, checkbox2, selectCheckbox);
                galleryImages.append(imageDiv);
                counter++;
            }
        }
        $("#submit-button").click(function() {
            if (fileList.length > 0) {
                const {formData, fileNames} = prepareFormData(fileList);
                let formDataArray = Array.from(formData.entries());

                if (formDataArray.length === 0) {
                    console.log('FormData is empty');
                } else {
                    console.log('FormData is not empty');
                }
                console.log('fileList:', fileList, 'fileNames:', fileNames);
                showLoadingSection();
                uploadImages(formData)
                    .then(data => handleUploadSuccess(data, fileList, fileNames))
                    .catch(error => {
                        console.error('Error uploading image', error);
                    })
            } else {
                alert("Please select at least one image for processing.");
            }
        });
    }).catch(error => {
        console.error('Error fetching gallery images:', error);
    });
}
async function selectImage(path, fileName) {
    const checkbox = document.getElementById(`select_${fileName}`);
    if (checkbox.checked) {
        const response = await fetch(`${protocol}://${hostAddress}:8000/${path}`);
        console.log(response)
        const data = await response.blob();
        const type = fileName.endsWith('.png') ? 'image/png' : 'image/jpeg';
        const file = new File([data], fileName, {type: type});
        fileList.push(file);
        selected_images.push(file);
    } else {
        const index = fileList.findIndex(file => file.name === fileName);
        if (index > -1) {
            fileList.splice(index, 1);
            selected_images.splice(index, 1);
        }
    }
    console.log(fileList);
}
function createImageElement(src, width, clickHandler) {
    const image = document.createElement("img");
    image.src = src;
    image.width = width;
    image.onclick = function() {
        clickHandler(src);
    };
    return image;
}

function createCheckboxElement(id, onchange, label) {
        const checkbox = document.createElement("div");
        checkbox.innerHTML = `<input type="checkbox" class="checkbox" id="${id}" onchange="${onchange}" name="${id}"><label class="check_label" for="${id}" >${label}</label>`;
        return checkbox;
    }

function enlargeImage(source) {
    let enlarged = $("#enlarged");
    enlarged.html(`<img src=${ source } height='80%' style='margin: 10px;'>`);
    enlarged.removeClass('hidden_tag');

    enlarged.click(function () {
        enlarged.addClass('hidden_tag');
    });
}

// async function enlarge_images() {
//     // Get the width and height of the #enlarged div
//     const enlargedDiv = document.querySelector('#enlarged');
//     const enlargedWidth = enlargedDiv.offsetWidth;
//     const enlargedHeight = enlargedDiv.offsetHeight;
//     // Get the dimensions of each image and find the largest one
//     const images = document.querySelectorAll('.enlarged-image');
//     let largestWidth = 0;
//     let largestHeight = 0;
//     let width = 0;
//     let height = 0;
//     await imageDimensions(images[0].src).then(dimensions => {
//         width = dimensions.width
//         height = dimensions.height
//
//         if (width > largestWidth) {
//             largestWidth = width;
//         }
//
//         if (height > largestHeight) {
//             largestHeight = height;
//         }
//
//         // Calculate the maximum size for the three images
//         const maxWidth = (enlargedWidth - 20) / 3;
//         const maxHeight = enlargedHeight - 20;
//
//         // Determine the width and height for each image
//         let imageWidth = largestWidth;
//         let imageHeight = largestHeight;
//
//         if (imageWidth > maxWidth) {
//             imageWidth = maxWidth;
//             imageHeight = (largestHeight / largestWidth) * maxWidth;
//         }
//
//         if (imageHeight > maxHeight) {
//             imageHeight = maxHeight;
//             imageWidth = (largestWidth / largestHeight) * maxHeight;
//         }
//         // Set the width and height for each image
//         images.forEach(image => {
//             image.style.width = imageWidth + 'px';
//             image.style.height = imageHeight + 'px';
//         });
//     })
// }
//
// const imageDimensions = file =>
//     new Promise((resolve, reject) => {
//         const img = new Image()
//
//         // the following handler will fire after a successful loading of the image
//         img.onload = () => {
//             const {naturalWidth: width, naturalHeight: height} = img
//             resolve({width, height})
//         }
//
//         // and this handler will fire if there was an error with the image (like if it's not really an image or a corrupted one)
//         img.onerror = () => {
//             reject('There was some problem with the image.')
//         }
//
//         img.src = file
//     })

async function downloadAllImages(user, protocol, host) {
    const images = document.querySelectorAll("#output-images img");
    const downloadPromises = [];
    console.log(images)
    for (let i = 0; i < images.length; i++) {
        const url = images[i].src;
        console.log(url)
        const filename = url.split("/").pop();
        console.log(filename)
        if (filename.includes('output')) {
            downloadPromises.push(downloadImage(url, filename));
        }
    }

    await Promise.all(downloadPromises)
        .then(() => {
            console.log('Temporary folder deleted successfully');
        })
        .catch(error => {
            console.error('Error deleting temporary folder', error);
        });

    // Reload the page
    reloadPage();
}

async function downloadImage(url, filename) {
    try {
        const response = await fetch(url);
        console.log('downloading image', filename)
        if (!response.ok) {
            throw new Error('Image not found or could not be downloaded.');
        }

        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (error) {
        console.error(error);
    }
}

async function deleteTempFolder(user_id, protocol, host) {
    try {
        console.log('deleting temp folder', user_id, protocol, host)
        const response = await fetch(`${protocol}://${host}:8000/delete/folder/`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({'X-User-Id': user_id})
        });

        if (response.ok) {
            console.log('Temporary folder deleted successfully');
        } else {
            console.error('Error deleting temporary folder');
        }
    } catch (error) {
        console.error('Error deleting temporary folder', error);
    }
}
