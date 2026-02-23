// Scale fix for mobile devices
(function(document) {
    'use strict';

    var metas = document.getElementsByTagName('meta'),
        changeViewportContent = function(content) {
            for (var i = 0; i < metas.length; i++) {
                if (metas[i].name === 'viewport') {
                    metas[i].content = content;
                    break;
                }
            }
        },
        initialize = function() {
            changeViewportContent('width=device-width, initial-scale=1.0, maximum-scale=1.0');
        },
        gestureStart = function() {
            changeViewportContent('width=device-width, initial-scale=1.0');
        },
        gestureEnd = function() {
            changeViewportContent('width=device-width, initial-scale=1.0, maximum-scale=1.0');
        };

    if (document.addEventListener) {
        document.addEventListener('DOMContentLoaded', initialize, false);
        document.addEventListener('gesturestart', gestureStart, false);
        document.addEventListener('gestureend', gestureEnd, false);
    }
}(document));
