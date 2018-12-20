$(function() {
    $('#mnist .input-image').on('change', function(event) {
        if(this.files && this.files[0]) {
            var reader = new FileReader();

            reader.onload = function(e) {
                $('#mnist .image-preview').attr('src', e.target.result);
            };

            reader.readAsDataURL(this.files[0]);

            $('#mnist .image-preview').show();

            var formData = new FormData();

            formData.append('image', this.files[0]);

            $.ajax({
                url: '/api/mnist/predict',
                type: 'POST',
                data: formData,
                success: function(data) {
                    $('#mnist .result .number-predicted').text(data.prediction.label);
                    $('#mnist .result').show();
                },
                cache: false,
                contentType: false,
                processData: false,
                xhr: function() { // Custom XMLHttpRequest
                    var myXhr = $.ajaxSettings.xhr();
    
                    if(myXhr.upload) { // Avalia se tem suporte a propriedade upload
                        myXhr.upload.addEventListener('progress', function() {
                            // faz alguma coisa durante o progresso do upload
                        }, false);
                    }
    
                    return myXhr;
                }
            });
        }
    });
});