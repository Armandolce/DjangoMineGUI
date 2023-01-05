const notificacionSwal=(titleText, text, icon, confirmationButtonText) => {
    Swal.fire({
        timer: 2000,
        timerProgressBar: false,
        title: titleText,
        text: text,
        icon: icon,
        showConfirmButton: false,
        confirmButtonText: confirmationButtonText,
    });
};