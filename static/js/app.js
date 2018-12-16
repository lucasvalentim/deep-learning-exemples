$(function() {
    $("#mnistForm").on('submit', function(event) {
        event.preventDefault();
        
        var formData = new FormData(this);
    
        $.ajax({
            url: '/api/mnist/predict',
            type: 'POST',
            data: formData,
            success: function(data) {
                console.log(data)
            },
            cache: false,
            contentType: false,
            processData: false,
            xhr: function() { // Custom XMLHttpRequest
                var myXhr = $.ajaxSettings.xhr();

                if(myXhr.upload) { // Avalia se tem suporte a propriedade upload
                    myXhr.upload.addEventListener('progress', function() {
                        /* faz alguma coisa durante o progresso do upload */
                    }, false);
                }

                return myXhr;
            }
        });
    });
});