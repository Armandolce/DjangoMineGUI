const notificacionSwal=(titleText, text, icon, confirmationButtonText) => {
    Swal.fire({
        timer: 5000,
        timerProgressBar: true,
        title: titleText,
        text: text,
        icon: icon,
        showConfirmButton: true,
        confirmButtonText: confirmationButtonText,
    });
};