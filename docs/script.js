/* =========================================================
   SIV-AI
   Frontend Interaction
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       BACKEND
    ===================================================== */

    const BACKEND_URL =
        "https://siv-ai-backend.onrender.com";


    /* =====================================================
       ELEMENTS
    ===================================================== */

    const uploadArea =
        document.getElementById("uploadArea");

    const imageInput =
        document.getElementById("imageInput");

    const uploadContent =
        document.getElementById("uploadContent");

    const inputPreview =
        document.getElementById("inputPreview");

    const inputFileName =
        document.getElementById("inputFileName");

    const restoreButton =
        document.getElementById("restoreButton");

    const outputPreview =
        document.getElementById("outputPreview");

    const outputPlaceholder =
        document.getElementById("outputPlaceholder");

    const waitingText =
        document.getElementById("waitingText");

    const loader =
        document.getElementById("loader");

    const downloadButton =
        document.getElementById("downloadButton");

    const resultPanel =
        document.getElementById("resultPanel");

    const psnrValue =
        document.getElementById("psnrValue");

    const ssimValue =
        document.getElementById("ssimValue");

    const runtimeValue =
        document.getElementById("runtimeValue");

    const feedbackForm =
        document.getElementById("feedbackForm");


    /* =====================================================
       STATE
    ===================================================== */

    let selectedFile = null;

    let restoredBlob = null;

    let restoredURL = null;


    /* =====================================================
       CHECK REQUIRED ELEMENTS
    ===================================================== */

    if (!uploadArea) {
        console.error("SIV-AI: uploadArea not found.");
        return;
    }

    if (!imageInput) {
        console.error("SIV-AI: imageInput not found.");
        return;
    }

    if (!restoreButton) {
        console.error("SIV-AI: restoreButton not found.");
        return;
    }


    /* =====================================================
       OPEN FILE SELECTOR
    ===================================================== */

    uploadArea.addEventListener(
        "click",
        () => {

            imageInput.click();

        }
    );


    /* =====================================================
       FILE SELECTED
    ===================================================== */

    imageInput.addEventListener(
        "change",
        (event) => {

            const file =
                event.target.files[0];

            if (file) {

                handleFile(file);

            }

        }
    );


    /* =====================================================
       DRAG ENTER
    ===================================================== */

    uploadArea.addEventListener(
        "dragover",
        (event) => {

            event.preventDefault();

            uploadArea.classList.add(
                "dragging"
            );

        }
    );


    /* =====================================================
       DRAG LEAVE
    ===================================================== */

    uploadArea.addEventListener(
        "dragleave",
        () => {

            uploadArea.classList.remove(
                "dragging"
            );

        }
    );


    /* =====================================================
       DROP
    ===================================================== */

    uploadArea.addEventListener(
        "drop",
        (event) => {

            event.preventDefault();

            uploadArea.classList.remove(
                "dragging"
            );

            const file =
                event.dataTransfer.files[0];

            if (file) {

                handleFile(file);

            }

        }
    );


    /* =====================================================
       HANDLE FILE
    ===================================================== */

    function handleFile(file) {

        selectedFile = file;


        /* -------------------------------------------------
           FILE NAME
        ------------------------------------------------- */

        if (inputFileName) {

            inputFileName.textContent =
                file.name.toUpperCase();

        }


        /* -------------------------------------------------
           ENABLE RESTORE
        ------------------------------------------------- */

        restoreButton.disabled = false;


        /* -------------------------------------------------
           IMAGE PREVIEW
        ------------------------------------------------- */

        if (
            file.type &&
            file.type.startsWith("image/")
        ) {

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

            reader.readAsDataURL(file);

        }

        else {

            /* ---------------------------------------------
               NPY FILE
            --------------------------------------------- */

            if (inputPreview) {

                inputPreview.classList.add(
                    "hidden"
                );

                inputPreview.src = "";

            }

            if (uploadContent) {

                uploadContent.classList.remove(
                    "hidden"
                );

                uploadContent.innerHTML = `
                    <div class="upload-icon">✓</div>
                    <strong>NPY FILE SELECTED</strong>
                    <span>${escapeHTML(file.name)}</span>
                `;

            }

        }


        /* -------------------------------------------------
           RESET OUTPUT
        ------------------------------------------------- */

        if (outputPreview) {

            outputPreview.classList.add(
                "hidden"
            );

            outputPreview.src = "";

        }


        if (outputPlaceholder) {

            outputPlaceholder.classList.remove(
                "hidden"
            );

        }


        if (waitingText) {

            waitingText.classList.remove(
                "hidden"
            );

        }


        if (loader) {

            loader.classList.add(
                "hidden"
            );

        }


        if (resultPanel) {

            resultPanel.classList.add(
                "hidden"
            );

        }


        if (downloadButton) {

            downloadButton.disabled = true;

        }


        restoredBlob = null;


        /* -------------------------------------------------
           RELEASE OLD OBJECT URL
        ------------------------------------------------- */

        if (restoredURL) {

            URL.revokeObjectURL(
                restoredURL
            );

            restoredURL = null;

        }

    }


    /* =====================================================
       RESTORE IMAGE
    ===================================================== */

    restoreButton.addEventListener(
        "click",
        async () => {

            /* ------------------------------------------------
               VALIDATE FILE
            ------------------------------------------------ */

            if (!selectedFile) {

                alert(
                    "Please upload an image first."
                );

                return;

            }


            /* ------------------------------------------------
               UI: LOADING
            ------------------------------------------------ */

            restoreButton.disabled = true;


            if (waitingText) {

                waitingText.classList.add(
                    "hidden"
                );

            }


            if (outputPlaceholder) {

                outputPlaceholder.classList.add(
                    "hidden"
                );

            }


            if (outputPreview) {

                outputPreview.classList.add(
                    "hidden"
                );

            }


            if (loader) {

                loader.classList.remove(
                    "hidden"
                );

            }


            if (resultPanel) {

                resultPanel.classList.add(
                    "hidden"
                );

            }


            if (downloadButton) {

                downloadButton.disabled = true;

            }


            /* ------------------------------------------------
               FORM DATA
            ------------------------------------------------ */

            const formData =
                new FormData();

            formData.append(
                "file",
                selectedFile
            );


            const start =
                performance.now();


            try {

                /* --------------------------------------------
                   SEND TO RENDER BACKEND
                -------------------------------------------- */

                console.log(
                    "Sending image to:",
                    `${BACKEND_URL}/api/restore`
                );


                const response =
                    await fetch(
                        `${BACKEND_URL}/api/restore`,
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                /* --------------------------------------------
                   HTTP ERROR
                -------------------------------------------- */

                if (!response.ok) {

                    let message =
                        `Restoration failed (${response.status}).`;


                    try {

                        const errorData =
                            await response.json();


                        if (
                            errorData &&
                            errorData.detail
                        ) {

                            message =
                                errorData.detail;

                        }

                    }

                    catch (_) {

                        /* Response was not JSON. */

                    }


                    throw new Error(
                        message
                    );

                }


                /* --------------------------------------------
                   RESPONSE HEADERS
                -------------------------------------------- */

                const runtimeHeader =
                    response.headers.get(
                        "X-SIV-AI-Runtime"
                    );


                const psnrHeader =
                    response.headers.get(
                        "X-SIV-AI-PSNR"
                    );


                const ssimHeader =
                    response.headers.get(
                        "X-SIV-AI-SSIM"
                    );


                const inputSizeHeader =
                    response.headers.get(
                        "X-SIV-AI-Input"
                    );


                const outputSizeHeader =
                    response.headers.get(
                        "X-SIV-AI-Output"
                    );


                console.log(
                    "PSNR header:",
                    psnrHeader
                );

                console.log(
                    "SSIM header:",
                    ssimHeader
                );

                console.log(
                    "Runtime header:",
                    runtimeHeader
                );


                /* --------------------------------------------
                   GET RESTORED PNG
                -------------------------------------------- */

                const blob =
                    await response.blob();


                if (
                    !blob ||
                    blob.size === 0
                ) {

                    throw new Error(
                        "Backend returned an empty image."
                    );

                }


                restoredBlob =
                    blob;


                /* --------------------------------------------
                   CREATE PREVIEW URL
                -------------------------------------------- */

                if (restoredURL) {

                    URL.revokeObjectURL(
                        restoredURL
                    );

                }


                restoredURL =
                    URL.createObjectURL(
                        blob
                    );


                if (outputPreview) {

                    outputPreview.src =
                        restoredURL;

                }


                /* --------------------------------------------
                   STOP LOADER
                -------------------------------------------- */

                if (loader) {

                    loader.classList.add(
                        "hidden"
                    );

                }


                /* --------------------------------------------
                   SHOW RESTORED IMAGE
                -------------------------------------------- */

                if (outputPreview) {

                    outputPreview.classList.remove(
                        "hidden"
                    );

                }


                /* --------------------------------------------
                   RUNTIME
                -------------------------------------------- */

                const measuredRuntime =
                    (
                        performance.now()
                        - start
                    ) / 1000;


                const runtime =
                    runtimeHeader ||
                    measuredRuntime.toFixed(4);


                /* --------------------------------------------
                   PSNR
                -------------------------------------------- */

                if (psnrValue) {

                    psnrValue.textContent =
                        psnrHeader
                            ? `${psnrHeader} dB`
                            : "N/A";

                }


                /* --------------------------------------------
                   SSIM
                -------------------------------------------- */

                if (ssimValue) {

                    ssimValue.textContent =
                        ssimHeader
                            ? ssimHeader
                            : "N/A";

                }


                /* --------------------------------------------
                   RUNTIME
                -------------------------------------------- */

                if (runtimeValue) {

                    runtimeValue.textContent =
                        `${runtime} s`;

                }


                /* --------------------------------------------
                   SHOW RESULT PANEL
                -------------------------------------------- */

                if (resultPanel) {

                    resultPanel.classList.remove(
                        "hidden"
                    );

                }


                /* --------------------------------------------
                   ENABLE DOWNLOAD
                -------------------------------------------- */

                if (downloadButton) {

                    downloadButton.disabled =
                        false;

                }


                /* --------------------------------------------
                   SUCCESS LOG
                -------------------------------------------- */

                console.log(
                    "===================================="
                );

                console.log(
                    "SIV-AI RESTORATION COMPLETE"
                );

                console.log(
                    "Backend:",
                    BACKEND_URL
                );

                console.log(
                    "Input:",
                    inputSizeHeader || "128x128"
                );

                console.log(
                    "Output:",
                    outputSizeHeader || "256x256"
                );

                console.log(
                    "PSNR:",
                    psnrHeader || "N/A"
                );

                console.log(
                    "SSIM:",
                    ssimHeader || "N/A"
                );

                console.log(
                    "Runtime:",
                    runtime,
                    "seconds"
                );

                console.log(
                    "===================================="
                );

            }


            catch (error) {

                console.error(
                    "SIV-AI restoration error:",
                    error
                );


                /* --------------------------------------------
                   RESET LOADING UI
                -------------------------------------------- */

                if (loader) {

                    loader.classList.add(
                        "hidden"
                    );

                }


                if (waitingText) {

                    waitingText.classList.remove(
                        "hidden"
                    );

                }


                if (outputPlaceholder) {

                    outputPlaceholder.classList.remove(
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


                /* --------------------------------------------
                   ERROR MESSAGE
                -------------------------------------------- */

                let errorMessage =
                    error.message ||
                    "Unknown error occurred.";


                if (
                    error instanceof TypeError
                ) {

                    errorMessage =
                        "Unable to connect to the SIV-AI backend.\n\n" +
                        "Please check whether the Render backend is running " +
                        "and CORS is enabled.";

                }


                alert(
                    "Restoration failed.\n\n" +
                    errorMessage
                );

            }


            finally {

                restoreButton.disabled =
                    false;

            }

        }
    );


    /* =====================================================
       DOWNLOAD RESTORED IMAGE
    ===================================================== */

    if (downloadButton) {

        downloadButton.addEventListener(
            "click",
            () => {

                if (!restoredBlob) {

                    return;

                }


                const url =
                    URL.createObjectURL(
                        restoredBlob
                    );


                const link =
                    document.createElement(
                        "a"
                    );


                link.href =
                    url;


                const originalName =
                    selectedFile
                        ? selectedFile.name
                        : "image";


                const baseName =
                    originalName.replace(
                        /\.[^/.]+$/,
                        ""
                    );


                link.download =
                    `${baseName}_SIV-AI_restored.png`;


                document.body.appendChild(
                    link
                );


                link.click();


                link.remove();


                setTimeout(
                    () => {

                        URL.revokeObjectURL(
                            url
                        );

                    },
                    1000
                );

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


            if (!question || !answer) {

                return;

            }


            question.addEventListener(
                "click",
                () => {

                    const isActive =
                        item.classList.contains(
                            "active"
                        );


                    /* -----------------------------------------
                       CLOSE ALL FAQ ITEMS
                    ----------------------------------------- */

                    faqItems.forEach(
                        (otherItem) => {

                            otherItem.classList.remove(
                                "active"
                            );


                            const otherAnswer =
                                otherItem.querySelector(
                                    ".faq-answer"
                                );


                            if (otherAnswer) {

                                otherAnswer.style.maxHeight =
                                    null;

                            }

                        }
                    );


                    /* -----------------------------------------
                       OPEN CLICKED ITEM
                    ----------------------------------------- */

                    if (!isActive) {

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
                                value <= selectedRating
                            );

                        }
                    );

                }
            );

        }
    );


    /* =====================================================
       FEEDBACK SUBMISSION
    ===================================================== */

    if (feedbackForm) {

        feedbackForm.addEventListener(
            "submit",
            async (event) => {

                event.preventDefault();


                const typeElement =
                    document.getElementById(
                        "feedbackType"
                    );


                const nameElement =
                    document.getElementById(
                        "feedbackName"
                    );


                const emailElement =
                    document.getElementById(
                        "feedbackEmail"
                    );


                const commentsElement =
                    document.getElementById(
                        "feedbackComments"
                    );


                const type =
                    typeElement
                        ? typeElement.value
                        : "General";


                const name =
                    nameElement
                        ? nameElement.value
                        : "";


                const email =
                    emailElement
                        ? emailElement.value
                        : "";


                const comments =
                    commentsElement
                        ? commentsElement.value
                        : "";


                try {

                    /* -----------------------------------------
                       SEND FEEDBACK TO RENDER
                    ----------------------------------------- */

                    const response =
                        await fetch(
                            `${BACKEND_URL}/api/feedback`,
                            {
                                method: "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify({

                                        type,

                                        name,

                                        email,

                                        rating:
                                            selectedRating || 5,

                                        comments

                                    })

                            }
                        );


                    /* -----------------------------------------
                       CHECK RESPONSE
                    ----------------------------------------- */

                    if (!response.ok) {

                        let message =
                            "Feedback submission failed.";


                        try {

                            const errorData =
                                await response.json();


                            if (
                                errorData &&
                                errorData.detail
                            ) {

                                message =
                                    errorData.detail;

                            }

                        }

                        catch (_) {

                            /* Ignore JSON parsing error. */

                        }


                        throw new Error(
                            message
                        );

                    }


                    /* -----------------------------------------
                       SUCCESS
                    ----------------------------------------- */

                    alert(
                        "Thank you! Your feedback has been submitted."
                    );


                    feedbackForm.reset();


                    selectedRating = 0;


                    ratingButtons.forEach(
                        (button) => {

                            button.classList.remove(
                                "selected"
                            );

                        }
                    );


                }


                catch (error) {

                    console.error(
                        "Feedback error:",
                        error
                    );


                    alert(
                        "Unable to submit feedback.\n\n" +
                        error.message
                    );

                }

            }
        );

    }


    /* =====================================================
       HTML ESCAPE
    ===================================================== */

    function escapeHTML(value) {

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
       CLEANUP
    ===================================================== */

    window.addEventListener(
        "beforeunload",
        () => {

            if (restoredURL) {

                URL.revokeObjectURL(
                    restoredURL
                );

                restoredURL = null;

            }

        }
    );


    /* =====================================================
       STARTUP LOG
    ===================================================== */

    console.log(
        "===================================="
    );

    console.log(
        "SIV-AI FRONTEND READY"
    );

    console.log(
        "Backend:",
        BACKEND_URL
    );

    console.log(
        "Restore endpoint:",
        `${BACKEND_URL}/api/restore`
    );

    console.log(
        "Feedback endpoint:",
        `${BACKEND_URL}/api/feedback`
    );

    console.log(
        "===================================="

    );

});