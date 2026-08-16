/* =========================================================
   SIV-AI
   Frontend Controller
   Upload + Restore + FAQ + Export
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    console.log("SIV-AI frontend initialized");


    /* =====================================================
       ELEMENTS
    ===================================================== */

    const uploadArea = document.getElementById("uploadArea");
    const imageInput = document.getElementById("imageInput");
    const uploadContent = document.getElementById("uploadContent");

    const inputPreview = document.getElementById("inputPreview");
    const outputPreview = document.getElementById("outputPreview");

    const inputFileName = document.getElementById("inputFileName");

    const restoreButton = document.getElementById("restoreButton");

    const downloadButton =
        document.getElementById("downloadButton");

    const loader =
        document.getElementById("loader");

    const waitingText =
        document.getElementById("waitingText");

    const resultPanel =
        document.getElementById("resultPanel");

    const psnrValue =
        document.getElementById("psnrValue");

    const ssimValue =
        document.getElementById("ssimValue");

    const runtimeValue =
        document.getElementById("runtimeValue");


    /* =====================================================
       STATE
    ===================================================== */

    let selectedFile = null;
    let restoredImageURL = null;


    /* =====================================================
       INITIAL STATE
    ===================================================== */

    if (restoreButton) {
        restoreButton.disabled = true;
    }

    if (downloadButton) {
        downloadButton.disabled = true;
    }


    /* =====================================================
       FILE TYPES
    ===================================================== */

    const supportedExtensions = [
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "npy"
    ];


    function isSupportedFile(file) {

        if (!file) {
            return false;
        }

        const name =
            file.name.toLowerCase();

        const extension =
            name.split(".").pop();

        return supportedExtensions.includes(
            extension
        );
    }


    /* =====================================================
       CLICK UPLOAD AREA
    ===================================================== */

    if (uploadArea && imageInput) {

        uploadArea.addEventListener(
            "click",
            (event) => {

                /*
                 * Prevent accidental second click
                 * when clicking an image.
                 */
                if (
                    event.target === imageInput
                ) {
                    return;
                }

                imageInput.click();
            }
        );


        imageInput.addEventListener(
            "change",
            () => {

                if (
                    imageInput.files &&
                    imageInput.files.length > 0
                ) {

                    handleFile(
                        imageInput.files[0]
                    );
                }
            }
        );
    }


    /* =====================================================
       DRAG ENTER
    ===================================================== */

    if (uploadArea) {

        uploadArea.addEventListener(
            "dragenter",
            (event) => {

                event.preventDefault();

                uploadArea.classList.add(
                    "dragging"
                );
            }
        );


        uploadArea.addEventListener(
            "dragover",
            (event) => {

                event.preventDefault();

                uploadArea.classList.add(
                    "dragging"
                );
            }
        );


        uploadArea.addEventListener(
            "dragleave",
            (event) => {

                event.preventDefault();

                uploadArea.classList.remove(
                    "dragging"
                );
            }
        );


        uploadArea.addEventListener(
            "drop",
            (event) => {

                event.preventDefault();

                uploadArea.classList.remove(
                    "dragging"
                );

                const files =
                    event.dataTransfer.files;

                if (
                    files &&
                    files.length > 0
                ) {

                    handleFile(files[0]);
                }
            }
        );
    }


    /* =====================================================
       HANDLE FILE
    ===================================================== */

    function handleFile(file) {

        console.log(
            "Selected file:",
            file.name,
            file.type,
            file.size
        );


        if (!isSupportedFile(file)) {

            alert(
                "Unsupported file format.\n\n" +
                "Supported formats:\n" +
                "PNG, JPG, JPEG, BMP, TIFF, WEBP, NPY"
            );

            return;
        }


        selectedFile = file;


        /* -------------------------------------------------
           SHOW FILE NAME
        ------------------------------------------------- */

        if (inputFileName) {

            inputFileName.textContent =
                file.name.toUpperCase();
        }


        /* -------------------------------------------------
           ENABLE RESTORE
        ------------------------------------------------- */

        if (restoreButton) {

            restoreButton.disabled = false;
        }


        /* -------------------------------------------------
           NPY FILE
        -------------------------------------------------

           Browser cannot display .npy directly.

           It can still be sent to FastAPI.
        ------------------------------------------------- */

        if (
            file.name
                .toLowerCase()
                .endsWith(".npy")
        ) {

            if (uploadContent) {

                uploadContent.classList.remove(
                    "hidden"
                );

                uploadContent.innerHTML = `
                    <div class="upload-icon">✓</div>
                    <strong>NPY INPUT READY</strong>
                    <span>${escapeHTML(file.name)}</span>
                `;
            }


            if (inputPreview) {

                inputPreview.classList.add(
                    "hidden"
                );
            }

            return;
        }


        /* -------------------------------------------------
           NORMAL IMAGE PREVIEW
        ------------------------------------------------- */

        const reader =
            new FileReader();


        reader.onload = (event) => {

            if (inputPreview) {

                inputPreview.src =
                    event.target.result;

                inputPreview.classList.remove(
                    "hidden"
                );
            }


            if (uploadContent) {

                uploadContent.classList.add(
                    "hidden"
                );
            }
        };


        reader.onerror = () => {

            alert(
                "Could not read the selected image."
            );
        };


        reader.readAsDataURL(file);
    }


    /* =====================================================
       RESTORE BUTTON
    ===================================================== */

    if (restoreButton) {

        restoreButton.addEventListener(
            "click",
            async () => {

                if (!selectedFile) {

                    alert(
                        "Please upload an SEM image first."
                    );

                    return;
                }


                await restoreImage();
            }
        );
    }


    /* =====================================================
       RESTORE IMAGE
    ===================================================== */

    async function restoreImage() {

        console.log(
            "Starting SIV-AI inference..."
        );


        /* -------------------------------------------------
           UI: PROCESSING
        ------------------------------------------------- */

        restoreButton.disabled = true;


        if (loader) {

            loader.classList.remove(
                "hidden"
            );
        }


        if (waitingText) {

            waitingText.classList.add(
                "hidden"
            );
        }


        if (outputPreview) {

            outputPreview.classList.add(
                "hidden"
            );
        }


        if (resultPanel) {

            resultPanel.classList.add(
                "hidden"
            );
        }


        const startTime =
            performance.now();


        try {

            const formData =
                new FormData();


            /*
             * IMPORTANT:
             * FastAPI normally expects:
             *
             * file: UploadFile
             *
             * Therefore field name is "file".
             */

            formData.append(
                "file",
                selectedFile
            );


            let response = null;
            let successfulEndpoint = null;


            /*
             * Try the common endpoint names.
             *
             * This makes the frontend tolerant of
             * your existing FastAPI route.
             */

            const endpoints = [
                "/restore",
                "/predict",
                "/inference"
            ];


            for (
                const endpoint
                of endpoints
            ) {

                console.log(
                    "Trying:",
                    endpoint
                );


                try {

                    const r =
                        await fetch(
                            endpoint,
                            {
                                method: "POST",
                                body: formData
                            }
                        );


                    if (r.ok) {

                        response = r;

                        successfulEndpoint =
                            endpoint;

                        break;
                    }


                    console.log(
                        endpoint,
                        "returned",
                        r.status
                    );

                } catch (endpointError) {

                    console.log(
                        endpoint,
                        "failed:",
                        endpointError
                    );
                }
            }


            if (!response) {

                throw new Error(
                    "FastAPI restoration endpoint could not be reached.\n\n" +
                    "Make sure uvicorn is running on port 8001."
                );
            }


            console.log(
                "Successful endpoint:",
                successfulEndpoint
            );


            const contentType =
                response.headers.get(
                    "content-type"
                ) || "";


            /* =================================================
               RESPONSE TYPE: IMAGE
            ================================================= */

            if (
                contentType.includes(
                    "image/"
                )
            ) {

                const blob =
                    await response.blob();


                displayRestoredImage(
                    blob
                );


                showRuntime(
                    startTime
                );


                showResultMetrics(
                    null
                );


                return;
            }


            /* =================================================
               RESPONSE TYPE: JSON
            ================================================= */

            if (
                contentType.includes(
                    "application/json"
                )
            ) {

                const data =
                    await response.json();


                console.log(
                    "Backend response:",
                    data
                );


                await processJSONResponse(
                    data,
                    startTime
                );


                return;
            }


            /* =================================================
               UNKNOWN RESPONSE
            ================================================= */

            const blob =
                await response.blob();


            if (
                blob.type &&
                blob.type.startsWith(
                    "image/"
                )
            ) {

                displayRestoredImage(
                    blob
                );

                showRuntime(
                    startTime
                );

                return;
            }


            throw new Error(
                "Backend returned an unsupported response."
            );


        } catch (error) {

            console.error(
                "SIV-AI restoration error:",
                error
            );


            alert(
                "Restoration failed.\n\n" +
                error.message
            );


        } finally {

            restoreButton.disabled =
                !selectedFile;


            if (loader) {

                loader.classList.add(
                    "hidden"
                );
            }
        }
    }


    /* =====================================================
       PROCESS JSON RESPONSE
    ===================================================== */

    async function processJSONResponse(
        data,
        startTime
    ) {

        console.log(
            "Processing backend JSON..."
        );


        /*
         * Possible image fields used by
         * different FastAPI implementations.
         */

        const imageFieldCandidates = [
            "image",
            "output",
            "result",
            "output_url",
            "image_url",
            "url",
            "restored_image",
            "restored_url"
        ];


        let imageValue = null;


        for (
            const field
            of imageFieldCandidates
        ) {

            if (
                data[field] !== undefined &&
                data[field] !== null
            ) {

                imageValue =
                    data[field];

                break;
            }
        }


        /* -------------------------------------------------
           DATA URL
        ------------------------------------------------- */

        if (
            typeof imageValue === "string" &&
            imageValue.startsWith(
                "data:image"
            )
        ) {

            if (outputPreview) {

                outputPreview.src =
                    imageValue;

                outputPreview.classList.remove(
                    "hidden"
                );
            }

            downloadButton.disabled = false;

            restoredImageURL =
                imageValue;

            showRuntime(startTime);

            showResultMetrics(data);

            return;
        }


        /* -------------------------------------------------
           BASE64
        ------------------------------------------------- */

        if (
            typeof imageValue === "string" &&
            looksLikeBase64Image(
                imageValue
            )
        ) {

            const dataURL =
                "data:image/png;base64," +
                imageValue;


            if (outputPreview) {

                outputPreview.src =
                    dataURL;

                outputPreview.classList.remove(
                    "hidden"
                );
            }


            restoredImageURL =
                dataURL;


            downloadButton.disabled =
                false;


            showRuntime(startTime);

            showResultMetrics(data);

            return;
        }


        /* -------------------------------------------------
           URL
        ------------------------------------------------- */

        if (
            typeof imageValue === "string"
        ) {

            if (outputPreview) {

                outputPreview.src =
                    imageValue;

                outputPreview.classList.remove(
                    "hidden"
                );
            }


            restoredImageURL =
                imageValue;


            downloadButton.disabled =
                false;


            showRuntime(startTime);

            showResultMetrics(data);

            return;
        }


        /*
         * Some APIs return:
         *
         * {
         *   "success": true,
         *   "output": "/outputs/result.png"
         * }
         *
         * handled above.
         */


        if (
            data.success === false
        ) {

            throw new Error(
                data.detail ||
                data.message ||
                "Backend restoration failed."
            );
        }


        /*
         * If the backend only returns metrics,
         * tell the user instead of silently failing.
         */

        showResultMetrics(data);

        showRuntime(startTime);


        throw new Error(
            "Backend completed but did not return a restored image."
        );
    }


    /* =====================================================
       DISPLAY RESTORED IMAGE
    ===================================================== */

    function displayRestoredImage(
        blob
    ) {

        if (!blob) {

            throw new Error(
                "Empty image returned by backend."
            );
        }


        if (restoredImageURL) {

            URL.revokeObjectURL(
                restoredImageURL
            );
        }


        restoredImageURL =
            URL.createObjectURL(blob);


        if (outputPreview) {

            outputPreview.src =
                restoredImageURL;

            outputPreview.classList.remove(
                "hidden"
            );
        }


        if (downloadButton) {

            downloadButton.disabled =
                false;
        }


        if (waitingText) {

            waitingText.classList.add(
                "hidden"
            );
        }


        console.log(
            "Restored image displayed."
        );
    }


    /* =====================================================
       RUNTIME
    ===================================================== */

    function showRuntime(
        startTime
    ) {

        if (!runtimeValue) {
            return;
        }


        const elapsed =
            performance.now() -
            startTime;


        runtimeValue.textContent =
            (elapsed / 1000).toFixed(2) +
            " s";
    }


    /* =====================================================
       METRICS
    ===================================================== */

    function showResultMetrics(
        data
    ) {

        if (!resultPanel) {
            return;
        }


        let psnr = null;
        let ssim = null;
        let runtime = null;


        if (data) {

            psnr =
                data.psnr ??
                data.PSNR ??
                data.psnr_value ??
                data.PSNR_dB;


            ssim =
                data.ssim ??
                data.SSIM ??
                data.ssim_value;


            runtime =
                data.runtime ??
                data.runtime_ms ??
                data.inference_time;
        }


        if (
            psnrValue &&
            psnr !== null &&
            psnr !== undefined
        ) {

            psnrValue.textContent =
                Number(psnr).toFixed(4) +
                " dB";
        }


        if (
            ssimValue &&
            ssim !== null &&
            ssim !== undefined
        ) {

            ssimValue.textContent =
                Number(ssim).toFixed(4);
        }


        if (
            runtimeValue &&
            runtime !== null &&
            runtime !== undefined
        ) {

            if (
                Number(runtime) > 100
            ) {

                runtimeValue.textContent =
                    Number(runtime).toFixed(0) +
                    " ms";

            } else {

                runtimeValue.textContent =
                    Number(runtime).toFixed(2) +
                    " s";
            }
        }


        resultPanel.classList.remove(
            "hidden"
        );
    }


    /* =====================================================
       DOWNLOAD
    ===================================================== */

    if (downloadButton) {

        downloadButton.addEventListener(
            "click",
            () => {

                if (!restoredImageURL) {

                    alert(
                        "No restored image is available."
                    );

                    return;
                }


                const link =
                    document.createElement(
                        "a"
                    );


                link.href =
                    restoredImageURL;


                link.download =
                    "SIV-AI_restored.png";


                document.body.appendChild(
                    link
                );


                link.click();


                link.remove();
            }
        );
    }


    /* =====================================================
       FAQ
    ===================================================== */

    const faqItems =
        document.querySelectorAll(
            ".faq-item"
        );


    faqItems.forEach(
        (item) => {

            const question =
                item.querySelector(
                    ".faq-question"
                );


            const answer =
                item.querySelector(
                    ".faq-answer"
                );


            if (
                !question ||
                !answer
            ) {
                return;
            }


            question.addEventListener(
                "click",
                () => {

                    const isActive =
                        item.classList.contains(
                            "active"
                        );


                    /*
                     * Close all other FAQs
                     */

                    faqItems.forEach(
                        (otherItem) => {

                            if (
                                otherItem !==
                                item
                            ) {

                                otherItem.classList.remove(
                                    "active"
                                );


                                const otherAnswer =
                                    otherItem.querySelector(
                                        ".faq-answer"
                                    );


                                if (
                                    otherAnswer
                                ) {

                                    otherAnswer.style.maxHeight =
                                        null;
                                }
                            }
                        }
                    );


                    /*
                     * Toggle selected FAQ
                     */

                    if (isActive) {

                        item.classList.remove(
                            "active"
                        );

                        answer.style.maxHeight =
                            null;

                    } else {

                        item.classList.add(
                            "active"
                        );

                        answer.style.maxHeight =
                            answer.scrollHeight +
                            "px";
                    }
                }
            );
        }
    );


    /* =====================================================
       FEEDBACK RATING
    ===================================================== */

    const ratingButtons =
        document.querySelectorAll(
            ".rating button"
        );


    let selectedRating = 0;


    ratingButtons.forEach(
        (button) => {

            button.addEventListener(
                "click",
                () => {

                    selectedRating =
                        Number(
                            button.dataset.rating
                        );


                    ratingButtons.forEach(
                        (btn) => {

                            const value =
                                Number(
                                    btn.dataset.rating
                                );


                            btn.classList.toggle(
                                "selected",
                                value <=
                                selectedRating
                            );
                        }
                    );
                }
            );
        }
    );


    /* =====================================================
       FEEDBACK FORM
    ===================================================== */

    const feedbackForm =
        document.getElementById(
            "feedbackForm"
        );


    if (feedbackForm) {

        feedbackForm.addEventListener(
            "submit",
            (event) => {

                event.preventDefault();


                const name =
                    document.getElementById(
                        "feedbackName"
                    )?.value.trim();


                const email =
                    document.getElementById(
                        "feedbackEmail"
                    )?.value.trim();


                const comments =
                    document.getElementById(
                        "feedbackComments"
                    )?.value.trim();


                console.log(
                    "SIV-AI Feedback:",
                    {
                        name,
                        email,
                        rating:
                            selectedRating,
                        comments
                    }
                );


                alert(
                    "Thank you for your feedback!"
                );


                feedbackForm.reset();


                ratingButtons.forEach(
                    (button) => {

                        button.classList.remove(
                            "selected"
                        );
                    }
                );


                selectedRating = 0;
            }
        );
    }


    /* =====================================================
       NAVIGATION
    ===================================================== */

    document.querySelectorAll(
        'a[href^="#"]'
    ).forEach(
        (link) => {

            link.addEventListener(
                "click",
                (event) => {

                    const targetID =
                        link.getAttribute(
                            "href"
                        );


                    if (
                        targetID === "#" ||
                        targetID.length <= 1
                    ) {
                        return;
                    }


                    const target =
                        document.querySelector(
                            targetID
                        );


                    if (!target) {
                        return;
                    }


                    event.preventDefault();


                    target.scrollIntoView({
                        behavior: "smooth",
                        block: "start"
                    });
                }
            );
        }
    );


    /* =====================================================
       HELPERS
    ===================================================== */

    function looksLikeBase64Image(
        value
    ) {

        if (
            typeof value !==
            "string"
        ) {
            return false;
        }


        return (
            value.length > 100 &&
            /^[A-Za-z0-9+/=]+$/.test(
                value
            )
        );
    }


    function escapeHTML(
        value
    ) {

        return String(value)
            .replace(
                /&/g,
                "&amp;"
            )
            .replace(
                /</g,
                "&lt;"
            )
            .replace(
                />/g,
                "&gt;"
            )
            .replace(
                /"/g,
                "&quot;"
            )
            .replace(
                /'/g,
                "&#039;"
            );
    }


    /* =====================================================
       READY
    ===================================================== */

    console.log(
        "SIV-AI UI ready."
    );

    console.log(
        "Upload:",
        !!uploadArea
    );

    console.log(
        "Restore:",
        !!restoreButton
    );

    console.log(
        "FAQ items:",
        faqItems.length
    );

});