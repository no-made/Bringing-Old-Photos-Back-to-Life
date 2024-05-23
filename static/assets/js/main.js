let selected_images = [];
let scratched_images = [];
let hd_images = [];
let fileNames = [];

$(document).ready(function () {
    const user_id = window.user_id;
    const port = window.port;
    const host = window.host;
    const hostAddress = window.location.hostname;
    const protocol = window.protocol;
    console.log('protocol: ', protocol, 'host: ', host, 'port: ', port, 'user_id: ', user_id)
    const landing_section = `
        <div id='landing'>
            <h1>Welcome to ReInHerit's Old Photos' Restorer!</h1>
            <p>Our virtual toolkit has been designed to assist museum owners and enthusiasts in effortlessly restoring old photos.
                <p>By simply uploading your photo, whether it has scratches or not, our advanced processing algorithms will work their magic.
                    Once the image processing is complete, you will have a fully restored photo to cherish and share with future generations.<br>
                    Try it out today and rediscover the beauty of your old photographs!</p>
            </p>
            <img id='cover_image' alt='landing image' src='static/assets/images/scratch_detection.png'>
            <a id='start_button' class='square_btn'>START TO RESTORE</a>
        </div>`;
    const input_section = `
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
                <div id='selected-images' ></div>
                <br>
                <a id='submit_label' class='square_btn' type='button' onclick="document.getElementById('submit-button').click(); return false;"> PROCESS </a>
                <button id='submit-button' type='submit' class='hidden_tag'></button>
                <br>
            </form>
            <div id='enlarged' class='hidden_tag'> </div>
        </div>`;

    const loading_div = `
        <div id='loader'>
            <p id='shadow'></p>
            <div id='shadow'></div>
            <div id='box'></div>
            <input type='text' id='loading-text' value='Starting process...' />
        </div>`;

    const output_section = `
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
            <br>
        </div>
        <div id='enlarged' class='hidden_tag'> </div>
    </div>`;

    const main_section = $("#central");
    main_section.html(landing_section);

    const startButton = $("#start_button");  // select the substitute button by ID
    const landing = $("#landing");  // select the landing element by ID

    // Function to replace content
    function replaceContent(content) {
        landing.replaceWith(content);
    }

    // Function to create image element
    function createImageElement(src, height, onclick) {
        const image = document.createElement("img");
        image.src = src;
        image.height = height;
        image.onclick = onclick;
        return image;
    }

    // Function to create checkbox element
    function createCheckboxElement(id, onchange, label) {
        const checkbox = document.createElement("div");
        checkbox.innerHTML = `<input type="checkbox" class="checkbox" id="${id}" onchange="${onchange}" name="${id}"><label class="check_label" for="${id}" >${label}</label>`;
        return checkbox;
    }

    // ENTERING INPUT SECTION
    startButton.click(function () {
        startButton.prop('disabled', true);
        replaceContent(input_section);  // replace the landing element with the new input section
        let files = {}
        let fileList = []

        const submit_label = $("#submit_label");  // select the landing element by ID
        submit_label.addClass('hidden_tag')

        $("#image-input").change(function () {  // add a change event listener to the file input
            const selected_files = $(this)[0].files;

            if (Object.keys(files).length === 0) {
                files = {...selected_files};
            } else {
                for (let i = 0; i < selected_files.length; i++) {
                    if (!Object.values(files).some(e =>
                        e.name === selected_files[i].name &&
                        e.size === selected_files[i].size &&
                        e.type === selected_files[i].type)) {
                        const new_key = Object.keys(files).length;
                        files = {...files, [new_key]: selected_files[i]};
                    }
                }
            }
            const dataTransfer = new DataTransfer();
            submit_label.removeClass('hidden_tag')

            for (let i = 0; i < Object.keys(files).length; i++) {
                const file = new File([files[i]], files[i].name, {
                    type: files[i].type,
                    lastModified: files[i].lastModified
                });
                dataTransfer.items.add(file);
            }
            fileList = dataTransfer.files;

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

                        const image = createImageElement(event.target.result, 200, function () {
                            let enlarged = $("#enlarged");
                            enlarged.html(`<img src='${image.src}' height='80%' style='margin: 10px;'>`);
                            enlarged.removeClass('hidden_tag');

                            enlarged.click(function () {
                                enlarged.addClass('hidden_tag');
                            });
                        });

                        const image_div = document.createElement("div");
                        image_div.id = `${image_name}_div`;

                        const checkbox_1 = createCheckboxElement(`check_${image_name}`, `checkboxChanged('${image_name}', '${i}', 'scratched')`, 'with scratches');
                        const checkbox_2 = createCheckboxElement(`check_hd_${image_name}`, `checkboxChanged('${image_name}', '${i}', 'hd')`, 'is HD');
                        checkbox_2.querySelector('input').disabled = true;
                        image_div.appendChild(image);
                        image_div.appendChild(checkbox_1);
                        image_div.appendChild(checkbox_2);

                        $("#selected-images").append(image_div);
                    };
                    // read the selected file as a data URL
                    reader.readAsDataURL(files[i]);
                }
            }
        })
        /* UPLOAD FILES */
        document.getElementById('image-form').addEventListener('submit', (event) => {
            event.preventDefault();  // prevent the default form submission
            fileNames = [];
            const files = fileList;

            for (let i = 0; i < files.length; i++) {
                const fileName = files[i].name;

                const scratched = scratched_images[i] !== '' ? 'true' : 'false';
                const hd = hd_images[i] !== '' ? 'true' : 'false';
                fileNames.push({'name': fileName, 'scratched': scratched, 'hd': hd});
            }
            console.log(fileNames)
        });

        // START TO PROCESS THEM ON THE SERVER
        document.getElementById('submit_label').addEventListener('click', (event) => {
            const formData = new FormData();
            const files = fileList;

            for (let i = 0; i < files.length; i++) {
                const fileName = files[i].name;

                const scratched = scratched_images[i] !== '' ? 'true' : 'false';
                const hd = hd_images[i] !== '' ? 'true' : 'false';
                formData.append('base', files[i]);
                formData.append('scratched', scratched);
                formData.append('hd', hd);
            }

            const input_section = $("#input-section");
            input_section.replaceWith(loading_div);

            const loading_text = document.getElementById('loading-text');
            const loading = $("#loader");
            const messages = ['Loading...', 'Please wait...', 'Almost done...', 'Hang tight...', 'Wow, it is quite large!', 'Oh my goodness!', 'What a tremendous size it is!'];
            // Get a reference to the loading text field
            let index = 0;
            setInterval(() => {
                loading_text.value = messages[index];
                index = (index + 1) % messages.length;
            }, 5000);
            fetch(`${protocol}://${hostAddress}:8000/upload/image/`, {
                method: 'POST',
                headers: {
                    'X-User-Id': user_id,
                    'Access-Control-Allow-Origin': '*'
                },
                body: formData,
                timeout: 65000 // timeout in milliseconds
            }).then(response => {
                if (response.ok) {
                    console.log('Image uploaded successfully');
                    return response.json()
                } else {
                    console.error('Error uploading image');
                }
            }).then(data => {
                loading.replaceWith(output_section);
                const output_images = $("#output-images");
                console.log(data)
                // Create an array to store the file names from data['images']
                const dataImageNames = data["images"].map(fileName => {
                    const file_name = fileName.split("\\").pop();
                    const name = file_name.split(".")[0];
                    return name;
                });
                console.log(dataImageNames)
                console.log(fileNames)
                // Loop through fileNames and check if each file is present in dataImageNames
                for (let i = 0; i < fileNames.length; i++) {
                    const fileName = fileNames[i].name;
                    const is_scratch = fileNames[i].scratched;
                    const is_hd = fileNames[i].hd;
                    const type = is_hd === 'true' ? '_hd_' : is_scratch === 'true' ? '_scratched_' : '_';
                    console.log('fileName: ', fileName, 'is_scratch: ', is_scratch, 'is_hd: ', is_hd)
                    const input_extension = fileName.split(".")[1];
                    console.log('input_extension: ', input_extension)
                    const ext = 'png';
                    const name = fileName.split(".")[0];
                    if (!dataImageNames.includes(name)) {
                        const folder_name = 'input' + type + 'images/';


                        const container = document.createElement("div");
                        container.classList.add("unavailable_container");
                        container.style.position = "relative";
                        container.style.display = "inline-block";
                        container.style.height = "200px";
                        container.style.margin = "10px";

                        const input_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${folder_name}${fileName}`, "100%", null);
                        input_image.style.filter = "grayscale(100%)";
                        input_image.style.height = "100%";

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

                        container.append(input_image);
                        container.append(textOverlay);
                        output_images.append(container);
                    } else {
                        // Image is present in data['images'], assign it to output_strips
                        const output_strips = document.createElement("div");
                        output_strips.classList.add("output_strips");

                        const fileBaseName = fileName.split(".")[0];
                        const fileExtension = fileName.split(".")[1];

                        const input_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${fileBaseName}_input.png`, 200, null);
                        const output_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${fileBaseName}_output.png`, 200, null);
                        const paragon_image = createImageElement(`${DJANGO_MEDIA_URL}${user_id}/${fileBaseName}_paragon.png`, 200, null);

                        output_strips.onclick = function () {
                            let enlarged = $("#enlarged");
                            enlarged.html(`<img src='${input_image.src}' class='enlarged-image'>
                                    <img src='${paragon_image.src}' class='enlarged-image'>
                                    <img src='${output_image.src}' class='enlarged-image'>`);

                            enlarge_images();
                            enlarged.removeClass('hidden_tag');
                            enlarged.click(function () {
                                enlarged.addClass('hidden_tag');
                            });
                        };

                        output_strips.append(input_image);
                        output_strips.append(paragon_image);
                        output_strips.append(output_image);
                        output_images.append(output_strips);
                    }
                }

                const download_button = document.getElementById("download_button");
                const restart_button = document.getElementById("restart_button");
                download_button.addEventListener("click", async function () {
                    await downloadAllImages(user_id, protocol, host);
                });
                restart_button.addEventListener("click", async function () {

                    reloadPage();
                });

            }).catch(error => {
                console.error('Error uploading image', error);
            }).finally(() => {
                // Re-enable the start button
                startButton.prop('disabled', false);
            });

        });

    });
    window.addEventListener('beforeunload', async function (event) {
        event.preventDefault()
        console.log('beforeunload')
        await deleteTempFolder(user_id, protocol, hostAddress);

    });

});

async function downloadAllImages(user, protocol, host) {
    const images = document.querySelectorAll("#output-images img");
    const downloadPromises = [];

    for (let i = 0; i < images.length; i++) {
        const url = images[i].src;
        const filename = url.split("/").pop();

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
        link.download = 'images/' + filename;
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
            body: JSON.stringify({ 'X-User-Id': user_id })
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

// Function to reload the page
function reloadPage() {
    location.reload();
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
    if (which === 'scratched') {
        if (scratched_images.indexOf(file_name) !== -1) {
            scratched_images[number] = ''
            console.log('removed: ', scratched_images)
        } else {
            scratched_images[number] = file_name
            console.log('added: ', scratched_images)
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
}

async function enlarge_images() {
    // Get the width and height of the #enlarged div
    const enlargedDiv = document.querySelector('#enlarged');
    const enlargedWidth = enlargedDiv.offsetWidth;
    const enlargedHeight = enlargedDiv.offsetHeight;
    // Get the dimensions of each image and find the largest one
    const images = document.querySelectorAll('.enlarged-image');
    let largestWidth = 0;
    let largestHeight = 0;
    let width = 0;
    let height = 0;
    await imageDimensions(images[0].src).then(dimensions => {
        width = dimensions.width
        height = dimensions.height

        if (width > largestWidth) {
            largestWidth = width;
        }

        if (height > largestHeight) {
            largestHeight = height;
        }

        // Calculate the maximum size for the three images
        const maxWidth = (enlargedWidth - 20) / 3;
        const maxHeight = enlargedHeight - 20;

        // Determine the width and height for each image
        let imageWidth = largestWidth;
        let imageHeight = largestHeight;

        if (imageWidth > maxWidth) {
            imageWidth = maxWidth;
            imageHeight = (largestHeight / largestWidth) * maxWidth;
        }

        if (imageHeight > maxHeight) {
            imageHeight = maxHeight;
            imageWidth = (largestWidth / largestHeight) * maxHeight;
        }
        // Set the width and height for each image
        images.forEach(image => {
            image.style.width = imageWidth + 'px';
            image.style.height = imageHeight + 'px';
        });
    })
}

// helper to get dimensions of an image
const imageDimensions = file =>
    new Promise((resolve, reject) => {
        const img = new Image()

        // the following handler will fire after a successful loading of the image
        img.onload = () => {
            const {naturalWidth: width, naturalHeight: height} = img
            resolve({width, height})
        }

        // and this handler will fire if there was an error with the image (like if it's not really an image or a corrupted one)
        img.onerror = () => {
            reject('There was some problem with the image.')
        }

        img.src = file
    })


