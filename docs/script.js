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
       OPEN FILE SELECTOR
    ===================================================== */

    uploadArea.addEventListener("click", () => {

        imageInput.click();

    });


    /* =====================================================
       FILE SELECTED
    ===================================================== */

    imageInput.addEventListener("change", (event) => {

        const file =
            event.target.files[0];

        if (file) {

            handleFile(file);

        }

    });


    /* =====================================================
       DRAG ENTER
    ===================================================== */

    uploadArea.addEventListener("dragover", (event) => {

        event.preventDefault();

        uploadArea.classList.add("dragging");

    });


    /* =====================================================
       DRAG LEAVE
    ===================================================== */

    uploadArea.addEventListener("dragleave", () => {

        uploadArea.classList.remove("dragging");

    });


    /* =====================================================
       DROP
    ===================================================== */

    uploadArea.addEventListener("drop", (event) => {

        event.preventDefault();

        uploadArea.classList.remove("dragging");

        const file =
            event.dataTransfer.files[0];

        if (file) {

            handleFile(file);

        }

    });


    /* =====================================================
       HANDLE FILE
    ===================================================== */

    function handleFile(file) {

        selectedFile = file;

        inputFileName.textContent =
            file.name.toUpperCase();

        restoreButton.disabled = false;


        /*
         * Browser preview for normal image files.
         */

        if (
            file.type.startsWith("image/")
        ) {

            const reader =
                new FileReader();

            reader.onload = (event) => {

                inputPreview.src =
                    event.target.result;

                inputPreview.classList.remove(
                    "hidden"
                );

                uploadContent.classList.add(
                    "hidden"
                );

            };

            reader.readAsDataURL(file);

        } else {

            /*
             * NPY file.
             */

            inputPreview.classList.add(
                "hidden"
            );

            uploadContent.classList.remove(
                "hidden"
            );

            uploadContent.innerHTML = `
                <div class="upload-icon">✓</div>
                <strong>NPY FILE SELECTED</strong>
                <span>${file.name}</span>
            `;

        }


        /*
         * Reset previous result.
         */

        outputPreview.classList.add(
            "hidden"
        );

        outputPreview.src = "";

        waitingText.classList.remove(
            "hidden"
        );

        loader.classList.add(
            "hidden"
        );

        resultPanel.classList.add(
            "hidden"
        );

        downloadButton.disabled = true;

        restoredBlob = null;


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

            if (!selectedFile) {

                alert(
                    "Please upload an image first."
                );

                return;

            }


            /* ---------------------------------------------
               UI: LOADING
            --------------------------------------------- */

            restoreButton.disabled = true;

            waitingText.classList.add(
                "hidden"
            );

            outputPreview.classList.add(
                "hidden"
            );

            loader.classList.remove(
                "hidden"
            );

            resultPanel.classList.add(
                "hidden"
            );

            downloadButton.disabled = true;


            /* ---------------------------------------------
               FORM DATA
            --------------------------------------------- */

            const formData =
                new FormData();

            formData.append(
                "file",
                selectedFile
            );


            const start =
                performance.now();


            try {

                /* -----------------------------------------
                   SEND TO RENDER FASTAPI BACKEND
                ----------------------------------------- */

                const response =
                    await fetch(
                        `${BACKEND_URL}/api/restore`,
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                /* -----------------------------------------
                   ERROR HANDLING
                ----------------------------------------- */

                if (!response.ok) {

                    let message =
                        "Restoration failed.";

                    try {

                        const errorData =
                            await response.json();

                        if (
                            errorData.detail
                        ) {

                            message =
                                errorData.detail;

                        }

                    } catch (_) {

                        /*
                         * Response was not JSON.
                         */

                    }

                    throw new Error(
                        message
                    );

                }


                /* -----------------------------------------
                   READ RESPONSE HEADERS
                ----------------------------------------- */

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


                /* -----------------------------------------
                   GET RESTORED IMAGE
                ----------------------------------------- */

                const blob =
                    await response.blob();

                restoredBlob =
                    blob;


                /* -----------------------------------------
                   CREATE PREVIEW URL
                ----------------------------------------- */

                if (restoredURL) {

                    URL.revokeObjectURL(
                        restoredURL
                    );

                }

                restoredURL =
                    URL.createObjectURL(
                        blob
                    );

                outputPreview.src =
                    restoredURL;


                /* -----------------------------------------
                   SHOW OUTPUT
                ----------------------------------------- */

                loader.classList.add(
                    "hidden"
                );

                outputPreview.classList.remove(
                    "hidden"
                );


                /* -----------------------------------------
                   METRICS
                ----------------------------------------- */

                const measuredRuntime =
                    (
                        performance.now()
                        - start
                    ) / 1000;


                const runtime =
                    runtimeHeader ||
                    measuredRuntime.toFixed(4);


                /*
                 * IMPORTANT:
                 * Do not fake PSNR/SSIM when the backend
                 * does not return them.
                 */

                psnrValue.textContent =
                    psnrHeader
                        ? `${psnrHeader} dB`
                        : "N/A";


                ssimValue.textContent =
                    ssimHeader
                        ? ssimHeader
                        : "N/A";


                runtimeValue.textContent =
                    `${runtime} s`;


                /* -----------------------------------------
                   SHOW RESULT PANEL
                ----------------------------------------- */

                resultPanel.classList.remove(
                    "hidden"
                );


                /* -----------------------------------------
                   ENABLE DOWNLOAD
                ----------------------------------------- */

                downloadButton.disabled =
                    false;


                /* -----------------------------------------
                   CONSOLE
                ----------------------------------------- */

                console.log(
                    "SIV-AI restoration complete"
                );

                console.log(
                    "Backend:",
                    BACKEND_URL
                );

                console.log(
                    "PSNR:",
                    psnrHeader
                );

                console.log(
                    "SSIM:",
                    ssimHeader
                );

                console.log(
                    "Runtime:",
                    runtime
                );


            } catch (error) {

                console.error(
                    "SIV-AI restoration error:",
                    error
                );


                loader.classList.add(
                    "hidden"
                );

                waitingText.classList.remove(
                    "hidden"
                );


                /*
                 * Give a more useful error message.
                 */

                let errorMessage =
                    error.message;


                if (
                    error instanceof TypeError
                ) {

                    errorMessage =
                        "Unable to connect to the SIV-AI backend. Please make sure the Render backend is running.";

                }


                alert(
                    "Restoration failed.\n\n"
                    + errorMessage
                );


            } finally {

                restoreButton.disabled =
                    false;

            }

        }
    );


    /* =====================================================
       DOWNLOAD RESTORED IMAGE
    ===================================================== */

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


            link.download =
                (
                    selectedFile
                        ? selectedFile.name
                            .replace(
                                /\.[^/.]+$/,
                                ""
                            )
                        : "image"
                )
                + "_SIV-AI_restored.png";


            document.body.appendChild(
                link
            );


            link.click();


            link.remove();


            setTimeout(() => {

                URL.revokeObjectURL(
                    url
                );

            }, 1000);

        }
    );


    /* =====================================================
       FAQ
    ===================================================== */

    const faqItems =
        document.querySelectorAll(
            ".faq-item"
        );


    faqItems.forEach((item) => {

        const question =
            item.querySelector(
                ".faq-question"
            );


        const answer =
            item.querySelector(
                ".faq-answer"
            );


        question.addEventListener(
            "click",
            () => {

                const isActive =
                    item.classList.contains(
                        "active"
                    );


                /*
                 * Close all FAQ items.
                 */

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


                /*
                 * Open clicked item.
                 */

                if (!isActive) {

                    item.classList.add(
                        "active"
                    );


                    answer.style.maxHeight =
                        answer.scrollHeight
                        + "px";

                }

            }
        );

    });


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


                const type =
                    document.getElementById(
                        "feedbackType"
                    ).value;


                const name =
                    document.getElementById(
                        "feedbackName"
                    ).value;


                const email =
                    document.getElementById(
                        "feedbackEmail"
                    ).value;


                const comments =
                    document.getElementById(
                        "feedbackComments"
                    ).value;


                try {

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
                                            selectedRating ||
                                            5,
                                        comments
                                    })
                            }
                        );


                    if (!response.ok) {

                        let message =
                            "Feedback submission failed.";

                        try {

                            const errorData =
                                await response.json();

                            if (
                                errorData.detail
                            ) {

                                message =
                                    errorData.detail;

                            }

                        } catch (_) {

                            /*
                             * Ignore JSON parsing error.
                             */

                        }

                        throw new Error(
                            message
                        );

                    }


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


                } catch (error) {

                    console.error(
                        "Feedback error:",
                        error
                    );


                    alert(
                        "Unable to submit feedback.\n\n"
                        + error.message
                    );

                }

            }
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

            }

        }
    );


});