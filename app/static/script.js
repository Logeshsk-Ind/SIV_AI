/* =========================================================
   SIV-AI
   Frontend Interaction
   GitHub Pages + Render FastAPI Backend
========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       BACKEND CONFIGURATION
    ===================================================== */

    /*
     * IMPORTANT
     *
     * The frontend is hosted on GitHub Pages.
     * The backend is hosted on Render.
     *
     * Therefore DO NOT use:
     *
     *     /api/restore
     *
     * because that would point to GitHub Pages.
     */

    const BACKEND_URL =
        "https://siv-ai-backend.onrender.com";


    /*
     * Maximum time allowed for a Render cold start.
     *
     * Render free services may sleep when inactive.
     * The first request can therefore take some time.
     */

    const BACKEND_TIMEOUT =
        120000;


    /*
     * Number of retry attempts.
     */

    const MAX_RETRIES =
        2;


    /*
     * Delay between retry attempts.
     */

    const RETRY_DELAY =
        3000;


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
       CHECK REQUIRED ELEMENTS
    ===================================================== */

    if (!uploadArea) {

        console.error(
            "SIV-AI: uploadArea element not found."
        );

        return;

    }


    if (!imageInput) {

        console.error(
            "SIV-AI: imageInput element not found."
        );

        return;

    }


    if (!restoreButton) {

        console.error(
            "SIV-AI: restoreButton element not found."
        );

        return;

    }


    /* =====================================================
       STATE
    ===================================================== */

    let selectedFile =
        null;

    let restoredBlob =
        null;

    let restoredURL =
        null;

    let selectedRating =
        0;


    /* =====================================================
       UTILITY
       SLEEP
    ===================================================== */

    function sleep(ms) {

        return new Promise(
            (resolve) => {
                setTimeout(
                    resolve,
                    ms
                );
            }
        );

    }


    /* =====================================================
       UTILITY
       FETCH WITH TIMEOUT
    ===================================================== */

    async function fetchWithTimeout(
        url,
        options = {},
        timeout = BACKEND_TIMEOUT
    ) {

        const controller =
            new AbortController();

        const timeoutId =
            setTimeout(
                () => {
                    controller.abort();
                },
                timeout
            );


        try {

            const response =
                await fetch(
                    url,
                    {
                        ...options,
                        signal:
                            controller.signal
                    }
                );

            return response;

        } finally {

            clearTimeout(
                timeoutId
            );

        }

    }


    /* =====================================================
       BACKEND HEALTH CHECK
    ===================================================== */

    async function checkBackend() {

        try {

            console.log(
                "Checking SIV-AI backend..."
            );

            const response =
                await fetchWithTimeout(
                    `${BACKEND_URL}/api/health`,
                    {
                        method: "GET",
                        cache: "no-store"
                    },
                    BACKEND_TIMEOUT
                );


            if (!response.ok) {

                throw new Error(
                    `Backend returned HTTP ${response.status}`
                );

            }


            const data =
                await response.json();


            console.log(
                "SIV-AI backend is online.",
                data
            );


            return data;


        } catch (error) {

            console.error(
                "Backend health check failed:",
                error
            );

            throw error;

        }

    }


    /* =====================================================
       BACKEND STATUS MESSAGE
    ===================================================== */

    function setBackendLoadingMessage() {

        if (waitingText) {

            waitingText.textContent =
                "CONNECTING TO SIV-AI AI ENGINE...";

        }

    }


    function setBackendWakingMessage() {

        if (waitingText) {

            waitingText.textContent =
                "STARTING SIV-AI AI ENGINE...";

        }

    }


    function setBackendReadyMessage() {

        if (waitingText) {

            waitingText.textContent =
                "AI ENGINE READY";

        }

    }


    function setWaitingMessage() {

        if (waitingText) {

            waitingText.textContent =
                "AWAITING INPUT";

        }

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

                handleFile(
                    file
                );

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

                handleFile(
                    file
                );

            }

        }
    );


    /* =====================================================
       HANDLE FILE
    ===================================================== */

    function handleFile(file) {

        selectedFile =
            file;


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

        restoreButton.disabled =
            false;


        /* -------------------------------------------------
           IMAGE PREVIEW
        ------------------------------------------------- */

        if (
            file.type &&
            file.type.startsWith("image/")
        ) {

            const reader =
                new FileReader();


            reader.onload =
                (event) => {

                    inputPreview.src =
                        event.target.result;


                    inputPreview.classList.remove(
                        "hidden"
                    );


                    if (uploadContent) {

                        uploadContent.classList.add(
                            "hidden"
                        );

                    }

                };


            reader.readAsDataURL(
                file
            );

        } else {

            /*
             * NPY file.
             *
             * Browsers cannot directly preview NPY.
             */

            inputPreview.classList.add(
                "hidden"
            );


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
           RESET PREVIOUS OUTPUT
        ------------------------------------------------- */

        outputPreview.classList.add(
            "hidden"
        );


        outputPreview.src =
            "";


        if (outputPlaceholder) {

            outputPlaceholder.classList.remove(
                "hidden"
            );

        }


        setWaitingMessage();


        loader.classList.add(
            "hidden"
        );


        resultPanel.classList.add(
            "hidden"
        );


        downloadButton.disabled =
            true;


        restoredBlob =
            null;


        /* -------------------------------------------------
           CLEAN OLD OBJECT URL
        ------------------------------------------------- */

        if (restoredURL) {

            URL.revokeObjectURL(
                restoredURL
            );


            restoredURL =
                null;

        }

    }


    /* =====================================================
       ESCAPE HTML
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
       READ ERROR RESPONSE
    ===================================================== */

    async function getErrorMessage(response) {

        let message =
            `Request failed with HTTP ${response.status}.`;


        try {

            const data =
                await response.json();


            if (
                data &&
                data.detail
            ) {

                message =
                    data.detail;

            }

        } catch (_) {

            /*
             * Response was not JSON.
             */

        }


        return message;

    }


    /* =====================================================
       RESTORE REQUEST
    ===================================================== */

    async function performRestore() {

        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        let lastError =
            null;


        for (
            let attempt = 1;
            attempt <= MAX_RETRIES + 1;
            attempt++
        ) {

            try {

                console.log(
                    `SIV-AI restore attempt ${attempt}`
                );


                const response =
                    await fetchWithTimeout(

                        `${BACKEND_URL}/api/restore`,

                        {
                            method: "POST",

                            body: formData,

                            cache: "no-store"
                        },

                        BACKEND_TIMEOUT

                    );


                if (!response.ok) {

                    const message =
                        await getErrorMessage(
                            response
                        );


                    throw new Error(
                        message
                    );

                }


                return response;


            } catch (error) {

                lastError =
                    error;


                console.error(
                    `Restore attempt ${attempt} failed:`,
                    error
                );


                /*
                 * If this was the final attempt,
                 * stop here.
                 */

                if (
                    attempt >
                    MAX_RETRIES
                ) {

                    break;

                }


                /*
                 * Tell the user that Render
                 * may be waking up.
                 */

                setBackendWakingMessage();


                await sleep(
                    RETRY_DELAY
                );

            }

        }


        throw lastError ||
            new Error(
                "Unable to connect to SIV-AI backend."
            );

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


            /* -------------------------------------------------
               UI: LOADING
            ------------------------------------------------- */

            restoreButton.disabled =
                true;


            downloadButton.disabled =
                true;


            waitingText.classList.add(
                "hidden"
            );


            outputPreview.classList.add(
                "hidden"
            );


            if (outputPlaceholder) {

                outputPlaceholder.classList.add(
                    "hidden"
                );

            }


            loader.classList.remove(
                "hidden"
            );


            resultPanel.classList.add(
                "hidden"
            );


            setBackendLoadingMessage();


            if (waitingText) {

                waitingText.classList.remove(
                    "hidden"
                );

            }


            /* -------------------------------------------------
               TIMER
            ------------------------------------------------- */

            const start =
                performance.now();


            try {

                /* ---------------------------------------------
                   STEP 1
                   Wake/check Render backend
                --------------------------------------------- */

                setBackendLoadingMessage();


                await checkBackend();


                setBackendReadyMessage();


                /*
                 * Small delay so the UI can update before
                 * the actual inference starts.
                 */

                await sleep(
                    250
                );


                /* ---------------------------------------------
                   STEP 2
                   RUN ACTUAL INFERENCE
                --------------------------------------------- */

                const response =
                    await performRestore();


                /* ---------------------------------------------
                   STEP 3
                   READ METRIC HEADERS
                --------------------------------------------- */

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


                const inputHeader =
                    response.headers.get(
                        "X-SIV-AI-Input"
                    );


                const outputHeader =
                    response.headers.get(
                        "X-SIV-AI-Output"
                    );


                console.log(
                    "SIV-AI headers:",
                    {
                        runtime:
                            runtimeHeader,

                        psnr:
                            psnrHeader,

                        ssim:
                            ssimHeader,

                        input:
                            inputHeader,

                        output:
                            outputHeader
                    }
                );


                /* ---------------------------------------------
                   STEP 4
                   READ RESTORED IMAGE
                --------------------------------------------- */

                const blob =
                    await response.blob();


                if (
                    !blob ||
                    blob.size === 0
                ) {

                    throw new Error(
                        "Backend returned an empty restored image."
                    );

                }


                restoredBlob =
                    blob;


                /* ---------------------------------------------
                   STEP 5
                   CREATE OUTPUT URL
                --------------------------------------------- */

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


                /* ---------------------------------------------
                   STEP 6
                   SHOW OUTPUT
                --------------------------------------------- */

                loader.classList.add(
                    "hidden"
                );


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


                outputPreview.classList.remove(
                    "hidden"
                );


                /* ---------------------------------------------
                   STEP 7
                   RUNTIME
                --------------------------------------------- */

                const measuredRuntime =
                    (
                        performance.now()
                        - start
                    ) / 1000;


                const runtime =
                    runtimeHeader ||
                    measuredRuntime.toFixed(4);


                /* ---------------------------------------------
                   STEP 8
                   PSNR
                --------------------------------------------- */

                /*
                 * IMPORTANT:
                 *
                 * We only display PSNR when the backend
                 * actually sends it.
                 *
                 * We do NOT fake it in JavaScript.
                 */

                if (psnrHeader) {

                    psnrValue.textContent =
                        `${psnrHeader} dB`;

                } else {

                    psnrValue.textContent =
                        "N/A";

                }


                /* ---------------------------------------------
                   STEP 9
                   SSIM
                --------------------------------------------- */

                if (ssimHeader) {

                    ssimValue.textContent =
                        ssimHeader;

                } else {

                    ssimValue.textContent =
                        "N/A";

                }


                /* ---------------------------------------------
                   STEP 10
                   RUNTIME DISPLAY
                --------------------------------------------- */

                runtimeValue.textContent =
                    `${runtime} s`;


                /* ---------------------------------------------
                   STEP 11
                   RESULT PANEL
                --------------------------------------------- */

                resultPanel.classList.remove(
                    "hidden"
                );


                /* ---------------------------------------------
                   STEP 12
                   DOWNLOAD
                --------------------------------------------- */

                downloadButton.disabled =
                    false;


                /* ---------------------------------------------
                   CONSOLE
                --------------------------------------------- */

                console.log(
                    "======================================"
                );

                console.log(
                    "SIV-AI RESTORATION COMPLETE"
                );

                console.log(
                    "======================================"
                );

                console.log(
                    "Backend:",
                    BACKEND_URL
                );

                console.log(
                    "Input:",
                    inputHeader
                );

                console.log(
                    "Output:",
                    outputHeader
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

                console.log(
                    "======================================"
                );


            } catch (error) {

                /* ---------------------------------------------
                   ERROR
                --------------------------------------------- */

                console.error(
                    "SIV-AI restoration error:",
                    error
                );


                loader.classList.add(
                    "hidden"
                );


                if (outputPlaceholder) {

                    outputPlaceholder.classList.remove(
                        "hidden"
                    );

                }


                outputPreview.classList.add(
                    "hidden"
                );


                if (waitingText) {

                    waitingText.classList.remove(
                        "hidden"
                    );

                    waitingText.textContent =
                        "BACKEND UNAVAILABLE";

                }


                resultPanel.classList.add(
                    "hidden"
                );


                let errorMessage =
                    "Unable to connect to the SIV-AI backend.";


                if (
                    error &&
                    error.name === "AbortError"
                ) {

                    errorMessage =
                        "The Render backend took too long to respond. It may be waking up. Please try again in a few seconds.";

                } else if (
                    error &&
                    error.message
                ) {

                    errorMessage =
                        error.message;

                }


                /*
                 * More useful message for GitHub Pages.
                 */

                alert(

                    "SIV-AI RESTORATION FAILED\n\n"

                    + errorMessage

                    + "\n\n"

                    + "Backend:\n"

                    + BACKEND_URL

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


            const baseName =
                selectedFile

                    ? selectedFile.name
                        .replace(
                            /\.[^/.]+$/,
                            ""
                        )

                    : "image";


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


                            if (
                                otherAnswer
                            ) {

                                otherAnswer.style.maxHeight =
                                    null;

                            }

                        }
                    );


                    /*
                     * Open selected FAQ.
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

        }
    );


    /* =====================================================
       FEEDBACK RATING
    ===================================================== */

    const ratingButtons =
        document.querySelectorAll(
            ".rating button"
        );


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

                    const response =
                        await fetchWithTimeout(

                            `${BACKEND_URL}/api/feedback`,

                            {

                                method:
                                    "POST",

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

                            },

                            BACKEND_TIMEOUT

                        );


                    if (!response.ok) {

                        const message =
                            await getErrorMessage(
                                response
                            );


                        throw new Error(
                            message
                        );

                    }


                    alert(
                        "Thank you! Your feedback has been submitted."
                    );


                    feedbackForm.reset();


                    selectedRating =
                        0;


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

                        + (
                            error.message ||
                            "Backend unavailable."
                        )

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


                restoredURL =
                    null;

            }

        }
    );


    /* =====================================================
       INITIAL BACKEND STATUS
    ===================================================== */

    /*
     * We intentionally DO NOT continuously ping Render.
     *
     * Render may sleep again after inactivity.
     *
     * The backend is checked when the user actually
     * presses RESTORE.
     */

    console.log(
        "======================================"
    );

    console.log(
        "SIV-AI FRONTEND READY"
    );

    console.log(
        "Backend:",
        BACKEND_URL
    );

    console.log(
        "Backend health:",
        `${BACKEND_URL}/api/health`
    );

    console.log(
        "======================================"

    );

});