/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import { ENGINE } from '../engine';
import { env } from '../environment';
import { FromPixels } from '../kernel_names';
import { getKernel } from '../kernel_registry';
import { Tensor } from '../tensor';
import { convertToTensor } from '../tensor_util_env';
import { cast } from './cast';
import { op } from './operation';
import { tensor3d } from './tensor3d';
let fromPixels2DContext;
/**
 * Creates a `tf.Tensor` from an image.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * tf.browser.fromPixels(image).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @returns A Tensor3D with the shape `[height, width, numChannels]`.
 *
 * Note: fromPixels can be lossy in some cases, same image may result in
 * slightly different tensor values, if rendered by different rendering
 * engines. This means that results from different browsers, or even same
 * browser with CPU and GPU rendering engines can be different. See discussion
 * in details:
 * https://github.com/tensorflow/tfjs/issues/5482
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
function fromPixels_(pixels, numChannels = 3) {
    // Sanity checks.
    if (numChannels > 4) {
        throw new Error('Cannot construct Tensor with more than 4 channels from pixels.');
    }
    if (pixels == null) {
        throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
    }
    let isPixelData = false;
    let isImageData = false;
    let isVideo = false;
    let isImage = false;
    let isCanvasLike = false;
    let isImageBitmap = false;
    if (pixels.data instanceof Uint8Array) {
        isPixelData = true;
    }
    else if (typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
        isImageData = true;
    }
    else if (typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement) {
        isVideo = true;
    }
    else if (typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement) {
        isImage = true;
        // tslint:disable-next-line: no-any
    }
    else if (pixels.getContext != null) {
        isCanvasLike = true;
    }
    else if (typeof (ImageBitmap) !== 'undefined' && pixels instanceof ImageBitmap) {
        isImageBitmap = true;
    }
    else {
        throw new Error('pixels passed to tf.browser.fromPixels() must be either an ' +
            `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
            `in browser, or OffscreenCanvas, ImageData in webworker` +
            ` or {data: Uint32Array, width: number, height: number}, ` +
            `but was ${pixels.constructor.name}`);
    }
    // If the current backend has 'FromPixels' registered, it has a more
    // efficient way of handling pixel uploads, so we call that.
    const kernel = getKernel(FromPixels, ENGINE.backendName);
    if (kernel != null) {
        const inputs = { pixels };
        const attrs = { numChannels };
        return ENGINE.runKernel(FromPixels, inputs, attrs);
    }
    const [width, height] = isVideo ?
        [
            pixels.videoWidth,
            pixels.videoHeight
        ] :
        [pixels.width, pixels.height];
    let vals;
    if (isCanvasLike) {
        vals =
            // tslint:disable-next-line:no-any
            pixels.getContext('2d').getImageData(0, 0, width, height).data;
    }
    else if (isImageData || isPixelData) {
        vals = pixels.data;
    }
    else if (isImage || isVideo || isImageBitmap) {
        if (fromPixels2DContext == null) {
            if (typeof document === 'undefined') {
                if (typeof OffscreenCanvas !== 'undefined' &&
                    typeof OffscreenCanvasRenderingContext2D !== 'undefined') {
                    // @ts-ignore
                    fromPixels2DContext = new OffscreenCanvas(1, 1).getContext('2d');
                }
                else {
                    throw new Error('Cannot parse input in current context. ' +
                        'Reason: OffscreenCanvas Context2D rendering is not supported.');
                }
            }
            else {
                fromPixels2DContext =
                    document.createElement('canvas').getContext('2d', { willReadFrequently: true });
            }
        }
        fromPixels2DContext.canvas.width = width;
        fromPixels2DContext.canvas.height = height;
        fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
        vals = fromPixels2DContext.getImageData(0, 0, width, height).data;
    }
    let values;
    if (numChannels === 4) {
        values = new Int32Array(vals);
    }
    else {
        const numPixels = width * height;
        values = new Int32Array(numPixels * numChannels);
        for (let i = 0; i < numPixels; i++) {
            for (let channel = 0; channel < numChannels; ++channel) {
                values[i * numChannels + channel] = vals[i * 4 + channel];
            }
        }
    }
    const outShape = [height, width, numChannels];
    return tensor3d(values, outShape, 'int32');
}
// Helper functions for |fromPixelsAsync| to check whether the input can
// be wrapped into imageBitmap.
function isPixelData(pixels) {
    return (pixels != null) && (pixels.data instanceof Uint8Array);
}
function isImageBitmapFullySupported() {
    return typeof window !== 'undefined' &&
        typeof (ImageBitmap) !== 'undefined' &&
        window.hasOwnProperty('createImageBitmap');
}
function isNonEmptyPixels(pixels) {
    return pixels != null && pixels.width !== 0 && pixels.height !== 0;
}
function canWrapPixelsToImageBitmap(pixels) {
    return isImageBitmapFullySupported() && !(pixels instanceof ImageBitmap) &&
        isNonEmptyPixels(pixels) && !isPixelData(pixels);
}
/**
 * Creates a `tf.Tensor` from an image in async way.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * (await tf.browser.fromPixelsAsync(image)).print();
 * ```
 * This API is the async version of fromPixels. The API will first
 * check |WRAP_TO_IMAGEBITMAP| flag, and try to wrap the input to
 * imageBitmap if the flag is set to true.
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
export async function fromPixelsAsync(pixels, numChannels = 3) {
    let inputs = null;
    // Check whether the backend needs to wrap |pixels| to imageBitmap and
    // whether |pixels| can be wrapped to imageBitmap.
    if (env().getBool('WRAP_TO_IMAGEBITMAP') &&
        canWrapPixelsToImageBitmap(pixels)) {
        // Force the imageBitmap creation to not do any premultiply alpha
        // ops.
        let imageBitmap;
        try {
            // wrap in try-catch block, because createImageBitmap may not work
            // properly in some browsers, e.g.
            // https://bugzilla.mozilla.org/show_bug.cgi?id=1335594
            // tslint:disable-next-line: no-any
            imageBitmap = await createImageBitmap(pixels, { premultiplyAlpha: 'none' });
        }
        catch (e) {
            imageBitmap = null;
        }
        // createImageBitmap will clip the source size.
        // In some cases, the input will have larger size than its content.
        // E.g. new Image(10, 10) but with 1 x 1 content. Using
        // createImageBitmap will clip the size from 10 x 10 to 1 x 1, which
        // is not correct. We should avoid wrapping such resouce to
        // imageBitmap.
        if (imageBitmap != null && imageBitmap.width === pixels.width &&
            imageBitmap.height === pixels.height) {
            inputs = imageBitmap;
        }
        else {
            inputs = pixels;
        }
    }
    else {
        inputs = pixels;
    }
    return fromPixels_(inputs, numChannels);
}
/**
 * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
 * canvas.
 *
 * When the dtype of the input is 'float32', we assume values in the range
 * [0-1]. Otherwise, when input is 'int32', we assume values in the range
 * [0-255].
 *
 * Returns a promise that resolves when the canvas has been drawn to.
 *
 * @param img A rank-2 tensor with shape `[height, width]`, or a rank-3 tensor
 * of shape `[height, width, numChannels]`. If rank-2, draws grayscale. If
 * rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 *
 * @doc {heading: 'Browser', namespace: 'browser'}
 */
export async function toPixels(img, canvas) {
    let $img = convertToTensor(img, 'img', 'toPixels');
    if (!(img instanceof Tensor)) {
        // Assume int32 if user passed a native array.
        const originalImgTensor = $img;
        $img = cast(originalImgTensor, 'int32');
        originalImgTensor.dispose();
    }
    if ($img.rank !== 2 && $img.rank !== 3) {
        throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.`);
    }
    const [height, width] = $img.shape.slice(0, 2);
    const depth = $img.rank === 2 ? 1 : $img.shape[2];
    if (depth > 4 || depth === 2) {
        throw new Error(`toPixels only supports depth of size ` +
            `1, 3 or 4 but got ${depth}`);
    }
    if ($img.dtype !== 'float32' && $img.dtype !== 'int32') {
        throw new Error(`Unsupported type for toPixels: ${$img.dtype}.` +
            ` Please use float32 or int32 tensors.`);
    }
    const data = await $img.data();
    const multiplier = $img.dtype === 'float32' ? 255 : 1;
    const bytes = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; ++i) {
        const rgba = [0, 0, 0, 255];
        for (let d = 0; d < depth; d++) {
            const value = data[i * depth + d];
            if ($img.dtype === 'float32') {
                if (value < 0 || value > 1) {
                    throw new Error(`Tensor values for a float32 Tensor must be in the ` +
                        `range [0 - 1] but encountered ${value}.`);
                }
            }
            else if ($img.dtype === 'int32') {
                if (value < 0 || value > 255) {
                    throw new Error(`Tensor values for a int32 Tensor must be in the ` +
                        `range [0 - 255] but encountered ${value}.`);
                }
            }
            if (depth === 1) {
                rgba[0] = value * multiplier;
                rgba[1] = value * multiplier;
                rgba[2] = value * multiplier;
            }
            else {
                rgba[d] = value * multiplier;
            }
        }
        const j = i * 4;
        bytes[j + 0] = Math.round(rgba[0]);
        bytes[j + 1] = Math.round(rgba[1]);
        bytes[j + 2] = Math.round(rgba[2]);
        bytes[j + 3] = Math.round(rgba[3]);
    }
    if (canvas != null) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(bytes, width, height);
        ctx.putImageData(imageData, 0, 0);
    }
    if ($img !== img) {
        $img.dispose();
    }
    return bytes;
}
export const fromPixels = /* @__PURE__ */ op({ fromPixels_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYnJvd3Nlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2Jyb3dzZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDbkMsT0FBTyxFQUFDLFVBQVUsRUFBb0MsTUFBTSxpQkFBaUIsQ0FBQztBQUM5RSxPQUFPLEVBQUMsU0FBUyxFQUFlLE1BQU0sb0JBQW9CLENBQUM7QUFDM0QsT0FBTyxFQUFDLE1BQU0sRUFBcUIsTUFBTSxXQUFXLENBQUM7QUFFckQsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBR25ELE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDNUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsUUFBUSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRXBDLElBQUksbUJBQTZDLENBQUM7QUFFbEQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0ErQkc7QUFDSCxTQUFTLFdBQVcsQ0FDaEIsTUFDNEIsRUFDNUIsV0FBVyxHQUFHLENBQUM7SUFDakIsaUJBQWlCO0lBQ2pCLElBQUksV0FBVyxHQUFHLENBQUMsRUFBRTtRQUNuQixNQUFNLElBQUksS0FBSyxDQUNYLGdFQUFnRSxDQUFDLENBQUM7S0FDdkU7SUFDRCxJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7UUFDbEIsTUFBTSxJQUFJLEtBQUssQ0FBQywwREFBMEQsQ0FBQyxDQUFDO0tBQzdFO0lBQ0QsSUFBSSxXQUFXLEdBQUcsS0FBSyxDQUFDO0lBQ3hCLElBQUksV0FBVyxHQUFHLEtBQUssQ0FBQztJQUN4QixJQUFJLE9BQU8sR0FBRyxLQUFLLENBQUM7SUFDcEIsSUFBSSxPQUFPLEdBQUcsS0FBSyxDQUFDO0lBQ3BCLElBQUksWUFBWSxHQUFHLEtBQUssQ0FBQztJQUN6QixJQUFJLGFBQWEsR0FBRyxLQUFLLENBQUM7SUFDMUIsSUFBSyxNQUFvQixDQUFDLElBQUksWUFBWSxVQUFVLEVBQUU7UUFDcEQsV0FBVyxHQUFHLElBQUksQ0FBQztLQUNwQjtTQUFNLElBQ0gsT0FBTyxDQUFDLFNBQVMsQ0FBQyxLQUFLLFdBQVcsSUFBSSxNQUFNLFlBQVksU0FBUyxFQUFFO1FBQ3JFLFdBQVcsR0FBRyxJQUFJLENBQUM7S0FDcEI7U0FBTSxJQUNILE9BQU8sQ0FBQyxnQkFBZ0IsQ0FBQyxLQUFLLFdBQVc7UUFDekMsTUFBTSxZQUFZLGdCQUFnQixFQUFFO1FBQ3RDLE9BQU8sR0FBRyxJQUFJLENBQUM7S0FDaEI7U0FBTSxJQUNILE9BQU8sQ0FBQyxnQkFBZ0IsQ0FBQyxLQUFLLFdBQVc7UUFDekMsTUFBTSxZQUFZLGdCQUFnQixFQUFFO1FBQ3RDLE9BQU8sR0FBRyxJQUFJLENBQUM7UUFDZixtQ0FBbUM7S0FDcEM7U0FBTSxJQUFLLE1BQWMsQ0FBQyxVQUFVLElBQUksSUFBSSxFQUFFO1FBQzdDLFlBQVksR0FBRyxJQUFJLENBQUM7S0FDckI7U0FBTSxJQUNILE9BQU8sQ0FBQyxXQUFXLENBQUMsS0FBSyxXQUFXLElBQUksTUFBTSxZQUFZLFdBQVcsRUFBRTtRQUN6RSxhQUFhLEdBQUcsSUFBSSxDQUFDO0tBQ3RCO1NBQU07UUFDTCxNQUFNLElBQUksS0FBSyxDQUNYLDZEQUE2RDtZQUM3RCxtRUFBbUU7WUFDbkUsd0RBQXdEO1lBQ3hELDBEQUEwRDtZQUMxRCxXQUFZLE1BQWEsQ0FBQyxXQUFXLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQztLQUNuRDtJQUNELG9FQUFvRTtJQUNwRSw0REFBNEQ7SUFDNUQsTUFBTSxNQUFNLEdBQUcsU0FBUyxDQUFDLFVBQVUsRUFBRSxNQUFNLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDekQsSUFBSSxNQUFNLElBQUksSUFBSSxFQUFFO1FBQ2xCLE1BQU0sTUFBTSxHQUFxQixFQUFDLE1BQU0sRUFBQyxDQUFDO1FBQzFDLE1BQU0sS0FBSyxHQUFvQixFQUFDLFdBQVcsRUFBQyxDQUFDO1FBQzdDLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsVUFBVSxFQUFFLE1BQW1DLEVBQy9DLEtBQWdDLENBQUMsQ0FBQztLQUN2QztJQUVELE1BQU0sQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUM7UUFDN0I7WUFDRyxNQUEyQixDQUFDLFVBQVU7WUFDdEMsTUFBMkIsQ0FBQyxXQUFXO1NBQ3pDLENBQUMsQ0FBQztRQUNILENBQUMsTUFBTSxDQUFDLEtBQUssRUFBRSxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDbEMsSUFBSSxJQUFrQyxDQUFDO0lBRXZDLElBQUksWUFBWSxFQUFFO1FBQ2hCLElBQUk7WUFDQSxrQ0FBa0M7WUFDakMsTUFBYyxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxZQUFZLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUMsSUFBSSxDQUFDO0tBQzdFO1NBQU0sSUFBSSxXQUFXLElBQUksV0FBVyxFQUFFO1FBQ3JDLElBQUksR0FBSSxNQUFnQyxDQUFDLElBQUksQ0FBQztLQUMvQztTQUFNLElBQUksT0FBTyxJQUFJLE9BQU8sSUFBSSxhQUFhLEVBQUU7UUFDOUMsSUFBSSxtQkFBbUIsSUFBSSxJQUFJLEVBQUU7WUFDL0IsSUFBSSxPQUFPLFFBQVEsS0FBSyxXQUFXLEVBQUU7Z0JBQ25DLElBQUksT0FBTyxlQUFlLEtBQUssV0FBVztvQkFDdEMsT0FBTyxpQ0FBaUMsS0FBSyxXQUFXLEVBQUU7b0JBQzVELGFBQWE7b0JBQ2IsbUJBQW1CLEdBQUcsSUFBSSxlQUFlLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQztpQkFDbEU7cUJBQU07b0JBQ0wsTUFBTSxJQUFJLEtBQUssQ0FDWCx5Q0FBeUM7d0JBQ3pDLCtEQUErRCxDQUFDLENBQUM7aUJBQ3RFO2FBQ0Y7aUJBQU07Z0JBQ0wsbUJBQW1CO29CQUNmLFFBQVEsQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLENBQUMsVUFBVSxDQUN2QyxJQUFJLEVBQUUsRUFBQyxrQkFBa0IsRUFBRSxJQUFJLEVBQUMsQ0FBQyxDQUFDO2FBQzNDO1NBQ0Y7UUFDRCxtQkFBbUIsQ0FBQyxNQUFNLENBQUMsS0FBSyxHQUFHLEtBQUssQ0FBQztRQUN6QyxtQkFBbUIsQ0FBQyxNQUFNLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztRQUMzQyxtQkFBbUIsQ0FBQyxTQUFTLENBQ3pCLE1BQTBCLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7UUFDckQsSUFBSSxHQUFHLG1CQUFtQixDQUFDLFlBQVksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQyxJQUFJLENBQUM7S0FDbkU7SUFDRCxJQUFJLE1BQWtCLENBQUM7SUFDdkIsSUFBSSxXQUFXLEtBQUssQ0FBQyxFQUFFO1FBQ3JCLE1BQU0sR0FBRyxJQUFJLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQztLQUMvQjtTQUFNO1FBQ0wsTUFBTSxTQUFTLEdBQUcsS0FBSyxHQUFHLE1BQU0sQ0FBQztRQUNqQyxNQUFNLEdBQUcsSUFBSSxVQUFVLENBQUMsU0FBUyxHQUFHLFdBQVcsQ0FBQyxDQUFDO1FBQ2pELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxTQUFTLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDbEMsS0FBSyxJQUFJLE9BQU8sR0FBRyxDQUFDLEVBQUUsT0FBTyxHQUFHLFdBQVcsRUFBRSxFQUFFLE9BQU8sRUFBRTtnQkFDdEQsTUFBTSxDQUFDLENBQUMsR0FBRyxXQUFXLEdBQUcsT0FBTyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUM7YUFDM0Q7U0FDRjtLQUNGO0lBQ0QsTUFBTSxRQUFRLEdBQTZCLENBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQztJQUN4RSxPQUFPLFFBQVEsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0FBQzdDLENBQUM7QUFFRCx3RUFBd0U7QUFDeEUsK0JBQStCO0FBQy9CLFNBQVMsV0FBVyxDQUFDLE1BRVc7SUFDOUIsT0FBTyxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsSUFBSSxDQUFFLE1BQW9CLENBQUMsSUFBSSxZQUFZLFVBQVUsQ0FBQyxDQUFDO0FBQ2hGLENBQUM7QUFFRCxTQUFTLDJCQUEyQjtJQUNsQyxPQUFPLE9BQU8sTUFBTSxLQUFLLFdBQVc7UUFDaEMsT0FBTyxDQUFDLFdBQVcsQ0FBQyxLQUFLLFdBQVc7UUFDcEMsTUFBTSxDQUFDLGNBQWMsQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDO0FBQ2pELENBQUM7QUFFRCxTQUFTLGdCQUFnQixDQUFDLE1BQzhDO0lBQ3RFLE9BQU8sTUFBTSxJQUFJLElBQUksSUFBSSxNQUFNLENBQUMsS0FBSyxLQUFLLENBQUMsSUFBSSxNQUFNLENBQUMsTUFBTSxLQUFLLENBQUMsQ0FBQztBQUNyRSxDQUFDO0FBRUQsU0FBUywwQkFBMEIsQ0FBQyxNQUU0QjtJQUM5RCxPQUFPLDJCQUEyQixFQUFFLElBQUksQ0FBQyxDQUFDLE1BQU0sWUFBWSxXQUFXLENBQUM7UUFDcEUsZ0JBQWdCLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUM7QUFDdkQsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBeUJHO0FBQ0gsTUFBTSxDQUFDLEtBQUssVUFBVSxlQUFlLENBQ2pDLE1BQzRCLEVBQzVCLFdBQVcsR0FBRyxDQUFDO0lBQ2pCLElBQUksTUFBTSxHQUN5QixJQUFJLENBQUM7SUFFeEMsc0VBQXNFO0lBQ3RFLGtEQUFrRDtJQUNsRCxJQUFJLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQyxxQkFBcUIsQ0FBQztRQUNwQywwQkFBMEIsQ0FBQyxNQUFNLENBQUMsRUFBRTtRQUN0QyxpRUFBaUU7UUFDakUsT0FBTztRQUNQLElBQUksV0FBVyxDQUFDO1FBRWhCLElBQUk7WUFDRixrRUFBa0U7WUFDbEUsa0NBQWtDO1lBQ2xDLHVEQUF1RDtZQUN2RCxtQ0FBbUM7WUFDbkMsV0FBVyxHQUFHLE1BQU8saUJBQXlCLENBQzFDLE1BQTJCLEVBQUUsRUFBQyxnQkFBZ0IsRUFBRSxNQUFNLEVBQUMsQ0FBQyxDQUFDO1NBQzlEO1FBQUMsT0FBTyxDQUFDLEVBQUU7WUFDVixXQUFXLEdBQUcsSUFBSSxDQUFDO1NBQ3BCO1FBRUQsK0NBQStDO1FBQy9DLG1FQUFtRTtRQUNuRSx1REFBdUQ7UUFDdkQsb0VBQW9FO1FBQ3BFLDJEQUEyRDtRQUMzRCxlQUFlO1FBQ2YsSUFBSSxXQUFXLElBQUksSUFBSSxJQUFJLFdBQVcsQ0FBQyxLQUFLLEtBQUssTUFBTSxDQUFDLEtBQUs7WUFDekQsV0FBVyxDQUFDLE1BQU0sS0FBSyxNQUFNLENBQUMsTUFBTSxFQUFFO1lBQ3hDLE1BQU0sR0FBRyxXQUFXLENBQUM7U0FDdEI7YUFBTTtZQUNMLE1BQU0sR0FBRyxNQUFNLENBQUM7U0FDakI7S0FDRjtTQUFNO1FBQ0wsTUFBTSxHQUFHLE1BQU0sQ0FBQztLQUNqQjtJQUVELE9BQU8sV0FBVyxDQUFDLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQztBQUMxQyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FtQkc7QUFDSCxNQUFNLENBQUMsS0FBSyxVQUFVLFFBQVEsQ0FDMUIsR0FBaUMsRUFDakMsTUFBMEI7SUFDNUIsSUFBSSxJQUFJLEdBQUcsZUFBZSxDQUFDLEdBQUcsRUFBRSxLQUFLLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFDbkQsSUFBSSxDQUFDLENBQUMsR0FBRyxZQUFZLE1BQU0sQ0FBQyxFQUFFO1FBQzVCLDhDQUE4QztRQUM5QyxNQUFNLGlCQUFpQixHQUFHLElBQUksQ0FBQztRQUMvQixJQUFJLEdBQUcsSUFBSSxDQUFDLGlCQUFpQixFQUFFLE9BQU8sQ0FBQyxDQUFDO1FBQ3hDLGlCQUFpQixDQUFDLE9BQU8sRUFBRSxDQUFDO0tBQzdCO0lBQ0QsSUFBSSxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUN0QyxNQUFNLElBQUksS0FBSyxDQUNYLHdEQUF3RCxJQUFJLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztLQUMzRTtJQUNELE1BQU0sQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQy9DLE1BQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFFbEQsSUFBSSxLQUFLLEdBQUcsQ0FBQyxJQUFJLEtBQUssS0FBSyxDQUFDLEVBQUU7UUFDNUIsTUFBTSxJQUFJLEtBQUssQ0FDWCx1Q0FBdUM7WUFDdkMscUJBQXFCLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDbkM7SUFFRCxJQUFJLElBQUksQ0FBQyxLQUFLLEtBQUssU0FBUyxJQUFJLElBQUksQ0FBQyxLQUFLLEtBQUssT0FBTyxFQUFFO1FBQ3RELE1BQU0sSUFBSSxLQUFLLENBQ1gsa0NBQWtDLElBQUksQ0FBQyxLQUFLLEdBQUc7WUFDL0MsdUNBQXVDLENBQUMsQ0FBQztLQUM5QztJQUVELE1BQU0sSUFBSSxHQUFHLE1BQU0sSUFBSSxDQUFDLElBQUksRUFBRSxDQUFDO0lBQy9CLE1BQU0sVUFBVSxHQUFHLElBQUksQ0FBQyxLQUFLLEtBQUssU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN0RCxNQUFNLEtBQUssR0FBRyxJQUFJLGlCQUFpQixDQUFDLEtBQUssR0FBRyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFFeEQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sR0FBRyxLQUFLLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDdkMsTUFBTSxJQUFJLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUU1QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsS0FBSyxFQUFFLENBQUMsRUFBRSxFQUFFO1lBQzlCLE1BQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxDQUFDLEdBQUcsS0FBSyxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBRWxDLElBQUksSUFBSSxDQUFDLEtBQUssS0FBSyxTQUFTLEVBQUU7Z0JBQzVCLElBQUksS0FBSyxHQUFHLENBQUMsSUFBSSxLQUFLLEdBQUcsQ0FBQyxFQUFFO29CQUMxQixNQUFNLElBQUksS0FBSyxDQUNYLG9EQUFvRDt3QkFDcEQsaUNBQWlDLEtBQUssR0FBRyxDQUFDLENBQUM7aUJBQ2hEO2FBQ0Y7aUJBQU0sSUFBSSxJQUFJLENBQUMsS0FBSyxLQUFLLE9BQU8sRUFBRTtnQkFDakMsSUFBSSxLQUFLLEdBQUcsQ0FBQyxJQUFJLEtBQUssR0FBRyxHQUFHLEVBQUU7b0JBQzVCLE1BQU0sSUFBSSxLQUFLLENBQ1gsa0RBQWtEO3dCQUNsRCxtQ0FBbUMsS0FBSyxHQUFHLENBQUMsQ0FBQztpQkFDbEQ7YUFDRjtZQUVELElBQUksS0FBSyxLQUFLLENBQUMsRUFBRTtnQkFDZixJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsS0FBSyxHQUFHLFVBQVUsQ0FBQztnQkFDN0IsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLEtBQUssR0FBRyxVQUFVLENBQUM7Z0JBQzdCLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLEdBQUcsVUFBVSxDQUFDO2FBQzlCO2lCQUFNO2dCQUNMLElBQUksQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLEdBQUcsVUFBVSxDQUFDO2FBQzlCO1NBQ0Y7UUFFRCxNQUFNLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLEtBQUssQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNuQyxLQUFLLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbkMsS0FBSyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLEtBQUssQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUNwQztJQUVELElBQUksTUFBTSxJQUFJLElBQUksRUFBRTtRQUNsQixNQUFNLENBQUMsS0FBSyxHQUFHLEtBQUssQ0FBQztRQUNyQixNQUFNLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztRQUN2QixNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsVUFBVSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ3BDLE1BQU0sU0FBUyxHQUFHLElBQUksU0FBUyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsTUFBTSxDQUFDLENBQUM7UUFDdEQsR0FBRyxDQUFDLFlBQVksQ0FBQyxTQUFTLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0tBQ25DO0lBQ0QsSUFBSSxJQUFJLEtBQUssR0FBRyxFQUFFO1FBQ2hCLElBQUksQ0FBQyxPQUFPLEVBQUUsQ0FBQztLQUNoQjtJQUNELE9BQU8sS0FBSyxDQUFDO0FBQ2YsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFVBQVUsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsV0FBVyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge2Vudn0gZnJvbSAnLi4vZW52aXJvbm1lbnQnO1xuaW1wb3J0IHtGcm9tUGl4ZWxzLCBGcm9tUGl4ZWxzQXR0cnMsIEZyb21QaXhlbHNJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge2dldEtlcm5lbCwgTmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjJELCBUZW5zb3IzRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7UGl4ZWxEYXRhLCBUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7Y2FzdH0gZnJvbSAnLi9jYXN0JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7dGVuc29yM2R9IGZyb20gJy4vdGVuc29yM2QnO1xuXG5sZXQgZnJvbVBpeGVsczJEQ29udGV4dDogQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEO1xuXG4vKipcbiAqIENyZWF0ZXMgYSBgdGYuVGVuc29yYCBmcm9tIGFuIGltYWdlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbWFnZSA9IG5ldyBJbWFnZURhdGEoMSwgMSk7XG4gKiBpbWFnZS5kYXRhWzBdID0gMTAwO1xuICogaW1hZ2UuZGF0YVsxXSA9IDE1MDtcbiAqIGltYWdlLmRhdGFbMl0gPSAyMDA7XG4gKiBpbWFnZS5kYXRhWzNdID0gMjU1O1xuICpcbiAqIHRmLmJyb3dzZXIuZnJvbVBpeGVscyhpbWFnZSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBwaXhlbHMgVGhlIGlucHV0IGltYWdlIHRvIGNvbnN0cnVjdCB0aGUgdGVuc29yIGZyb20uIFRoZVxuICogc3VwcG9ydGVkIGltYWdlIHR5cGVzIGFyZSBhbGwgNC1jaGFubmVsLiBZb3UgY2FuIGFsc28gcGFzcyBpbiBhbiBpbWFnZVxuICogb2JqZWN0IHdpdGggZm9sbG93aW5nIGF0dHJpYnV0ZXM6XG4gKiBge2RhdGE6IFVpbnQ4QXJyYXk7IHdpZHRoOiBudW1iZXI7IGhlaWdodDogbnVtYmVyfWBcbiAqIEBwYXJhbSBudW1DaGFubmVscyBUaGUgbnVtYmVyIG9mIGNoYW5uZWxzIG9mIHRoZSBvdXRwdXQgdGVuc29yLiBBXG4gKiBudW1DaGFubmVscyB2YWx1ZSBsZXNzIHRoYW4gNCBhbGxvd3MgeW91IHRvIGlnbm9yZSBjaGFubmVscy4gRGVmYXVsdHMgdG9cbiAqIDMgKGlnbm9yZXMgYWxwaGEgY2hhbm5lbCBvZiBpbnB1dCBpbWFnZSkuXG4gKlxuICogQHJldHVybnMgQSBUZW5zb3IzRCB3aXRoIHRoZSBzaGFwZSBgW2hlaWdodCwgd2lkdGgsIG51bUNoYW5uZWxzXWAuXG4gKlxuICogTm90ZTogZnJvbVBpeGVscyBjYW4gYmUgbG9zc3kgaW4gc29tZSBjYXNlcywgc2FtZSBpbWFnZSBtYXkgcmVzdWx0IGluXG4gKiBzbGlnaHRseSBkaWZmZXJlbnQgdGVuc29yIHZhbHVlcywgaWYgcmVuZGVyZWQgYnkgZGlmZmVyZW50IHJlbmRlcmluZ1xuICogZW5naW5lcy4gVGhpcyBtZWFucyB0aGF0IHJlc3VsdHMgZnJvbSBkaWZmZXJlbnQgYnJvd3NlcnMsIG9yIGV2ZW4gc2FtZVxuICogYnJvd3NlciB3aXRoIENQVSBhbmQgR1BVIHJlbmRlcmluZyBlbmdpbmVzIGNhbiBiZSBkaWZmZXJlbnQuIFNlZSBkaXNjdXNzaW9uXG4gKiBpbiBkZXRhaWxzOlxuICogaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9pc3N1ZXMvNTQ4MlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdCcm93c2VyJywgbmFtZXNwYWNlOiAnYnJvd3NlcicsIGlnbm9yZUNJOiB0cnVlfVxuICovXG5mdW5jdGlvbiBmcm9tUGl4ZWxzXyhcbiAgICBwaXhlbHM6IFBpeGVsRGF0YXxJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxIVE1MQ2FudmFzRWxlbWVudHxcbiAgICBIVE1MVmlkZW9FbGVtZW50fEltYWdlQml0bWFwLFxuICAgIG51bUNoYW5uZWxzID0gMyk6IFRlbnNvcjNEIHtcbiAgLy8gU2FuaXR5IGNoZWNrcy5cbiAgaWYgKG51bUNoYW5uZWxzID4gNCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ0Nhbm5vdCBjb25zdHJ1Y3QgVGVuc29yIHdpdGggbW9yZSB0aGFuIDQgY2hhbm5lbHMgZnJvbSBwaXhlbHMuJyk7XG4gIH1cbiAgaWYgKHBpeGVscyA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdwaXhlbHMgcGFzc2VkIHRvIHRmLmJyb3dzZXIuZnJvbVBpeGVscygpIGNhbiBub3QgYmUgbnVsbCcpO1xuICB9XG4gIGxldCBpc1BpeGVsRGF0YSA9IGZhbHNlO1xuICBsZXQgaXNJbWFnZURhdGEgPSBmYWxzZTtcbiAgbGV0IGlzVmlkZW8gPSBmYWxzZTtcbiAgbGV0IGlzSW1hZ2UgPSBmYWxzZTtcbiAgbGV0IGlzQ2FudmFzTGlrZSA9IGZhbHNlO1xuICBsZXQgaXNJbWFnZUJpdG1hcCA9IGZhbHNlO1xuICBpZiAoKHBpeGVscyBhcyBQaXhlbERhdGEpLmRhdGEgaW5zdGFuY2VvZiBVaW50OEFycmF5KSB7XG4gICAgaXNQaXhlbERhdGEgPSB0cnVlO1xuICB9IGVsc2UgaWYgKFxuICAgICAgdHlwZW9mIChJbWFnZURhdGEpICE9PSAndW5kZWZpbmVkJyAmJiBwaXhlbHMgaW5zdGFuY2VvZiBJbWFnZURhdGEpIHtcbiAgICBpc0ltYWdlRGF0YSA9IHRydWU7XG4gIH0gZWxzZSBpZiAoXG4gICAgICB0eXBlb2YgKEhUTUxWaWRlb0VsZW1lbnQpICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICAgcGl4ZWxzIGluc3RhbmNlb2YgSFRNTFZpZGVvRWxlbWVudCkge1xuICAgIGlzVmlkZW8gPSB0cnVlO1xuICB9IGVsc2UgaWYgKFxuICAgICAgdHlwZW9mIChIVE1MSW1hZ2VFbGVtZW50KSAhPT0gJ3VuZGVmaW5lZCcgJiZcbiAgICAgIHBpeGVscyBpbnN0YW5jZW9mIEhUTUxJbWFnZUVsZW1lbnQpIHtcbiAgICBpc0ltYWdlID0gdHJ1ZTtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWFueVxuICB9IGVsc2UgaWYgKChwaXhlbHMgYXMgYW55KS5nZXRDb250ZXh0ICE9IG51bGwpIHtcbiAgICBpc0NhbnZhc0xpa2UgPSB0cnVlO1xuICB9IGVsc2UgaWYgKFxuICAgICAgdHlwZW9mIChJbWFnZUJpdG1hcCkgIT09ICd1bmRlZmluZWQnICYmIHBpeGVscyBpbnN0YW5jZW9mIEltYWdlQml0bWFwKSB7XG4gICAgaXNJbWFnZUJpdG1hcCA9IHRydWU7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAncGl4ZWxzIHBhc3NlZCB0byB0Zi5icm93c2VyLmZyb21QaXhlbHMoKSBtdXN0IGJlIGVpdGhlciBhbiAnICtcbiAgICAgICAgYEhUTUxWaWRlb0VsZW1lbnQsIEhUTUxJbWFnZUVsZW1lbnQsIEhUTUxDYW52YXNFbGVtZW50LCBJbWFnZURhdGEgYCArXG4gICAgICAgIGBpbiBicm93c2VyLCBvciBPZmZzY3JlZW5DYW52YXMsIEltYWdlRGF0YSBpbiB3ZWJ3b3JrZXJgICtcbiAgICAgICAgYCBvciB7ZGF0YTogVWludDMyQXJyYXksIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyfSwgYCArXG4gICAgICAgIGBidXQgd2FzICR7KHBpeGVscyBhcyB7fSkuY29uc3RydWN0b3IubmFtZX1gKTtcbiAgfVxuICAvLyBJZiB0aGUgY3VycmVudCBiYWNrZW5kIGhhcyAnRnJvbVBpeGVscycgcmVnaXN0ZXJlZCwgaXQgaGFzIGEgbW9yZVxuICAvLyBlZmZpY2llbnQgd2F5IG9mIGhhbmRsaW5nIHBpeGVsIHVwbG9hZHMsIHNvIHdlIGNhbGwgdGhhdC5cbiAgY29uc3Qga2VybmVsID0gZ2V0S2VybmVsKEZyb21QaXhlbHMsIEVOR0lORS5iYWNrZW5kTmFtZSk7XG4gIGlmIChrZXJuZWwgIT0gbnVsbCkge1xuICAgIGNvbnN0IGlucHV0czogRnJvbVBpeGVsc0lucHV0cyA9IHtwaXhlbHN9O1xuICAgIGNvbnN0IGF0dHJzOiBGcm9tUGl4ZWxzQXR0cnMgPSB7bnVtQ2hhbm5lbHN9O1xuICAgIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICBGcm9tUGl4ZWxzLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xuICB9XG5cbiAgY29uc3QgW3dpZHRoLCBoZWlnaHRdID0gaXNWaWRlbyA/XG4gICAgICBbXG4gICAgICAgIChwaXhlbHMgYXMgSFRNTFZpZGVvRWxlbWVudCkudmlkZW9XaWR0aCxcbiAgICAgICAgKHBpeGVscyBhcyBIVE1MVmlkZW9FbGVtZW50KS52aWRlb0hlaWdodFxuICAgICAgXSA6XG4gICAgICBbcGl4ZWxzLndpZHRoLCBwaXhlbHMuaGVpZ2h0XTtcbiAgbGV0IHZhbHM6IFVpbnQ4Q2xhbXBlZEFycmF5fFVpbnQ4QXJyYXk7XG5cbiAgaWYgKGlzQ2FudmFzTGlrZSkge1xuICAgIHZhbHMgPVxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgIChwaXhlbHMgYXMgYW55KS5nZXRDb250ZXh0KCcyZCcpLmdldEltYWdlRGF0YSgwLCAwLCB3aWR0aCwgaGVpZ2h0KS5kYXRhO1xuICB9IGVsc2UgaWYgKGlzSW1hZ2VEYXRhIHx8IGlzUGl4ZWxEYXRhKSB7XG4gICAgdmFscyA9IChwaXhlbHMgYXMgUGl4ZWxEYXRhIHwgSW1hZ2VEYXRhKS5kYXRhO1xuICB9IGVsc2UgaWYgKGlzSW1hZ2UgfHwgaXNWaWRlbyB8fCBpc0ltYWdlQml0bWFwKSB7XG4gICAgaWYgKGZyb21QaXhlbHMyRENvbnRleHQgPT0gbnVsbCkge1xuICAgICAgaWYgKHR5cGVvZiBkb2N1bWVudCA9PT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgaWYgKHR5cGVvZiBPZmZzY3JlZW5DYW52YXMgIT09ICd1bmRlZmluZWQnICYmXG4gICAgICAgICAgICB0eXBlb2YgT2Zmc2NyZWVuQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEICE9PSAndW5kZWZpbmVkJykge1xuICAgICAgICAgIC8vIEB0cy1pZ25vcmVcbiAgICAgICAgICBmcm9tUGl4ZWxzMkRDb250ZXh0ID0gbmV3IE9mZnNjcmVlbkNhbnZhcygxLCAxKS5nZXRDb250ZXh0KCcyZCcpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgJ0Nhbm5vdCBwYXJzZSBpbnB1dCBpbiBjdXJyZW50IGNvbnRleHQuICcgK1xuICAgICAgICAgICAgICAnUmVhc29uOiBPZmZzY3JlZW5DYW52YXMgQ29udGV4dDJEIHJlbmRlcmluZyBpcyBub3Qgc3VwcG9ydGVkLicpO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBmcm9tUGl4ZWxzMkRDb250ZXh0ID1cbiAgICAgICAgICAgIGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpLmdldENvbnRleHQoXG4gICAgICAgICAgICAgICAgJzJkJywge3dpbGxSZWFkRnJlcXVlbnRseTogdHJ1ZX0pO1xuICAgICAgfVxuICAgIH1cbiAgICBmcm9tUGl4ZWxzMkRDb250ZXh0LmNhbnZhcy53aWR0aCA9IHdpZHRoO1xuICAgIGZyb21QaXhlbHMyRENvbnRleHQuY2FudmFzLmhlaWdodCA9IGhlaWdodDtcbiAgICBmcm9tUGl4ZWxzMkRDb250ZXh0LmRyYXdJbWFnZShcbiAgICAgICAgcGl4ZWxzIGFzIEhUTUxWaWRlb0VsZW1lbnQsIDAsIDAsIHdpZHRoLCBoZWlnaHQpO1xuICAgIHZhbHMgPSBmcm9tUGl4ZWxzMkRDb250ZXh0LmdldEltYWdlRGF0YSgwLCAwLCB3aWR0aCwgaGVpZ2h0KS5kYXRhO1xuICB9XG4gIGxldCB2YWx1ZXM6IEludDMyQXJyYXk7XG4gIGlmIChudW1DaGFubmVscyA9PT0gNCkge1xuICAgIHZhbHVlcyA9IG5ldyBJbnQzMkFycmF5KHZhbHMpO1xuICB9IGVsc2Uge1xuICAgIGNvbnN0IG51bVBpeGVscyA9IHdpZHRoICogaGVpZ2h0O1xuICAgIHZhbHVlcyA9IG5ldyBJbnQzMkFycmF5KG51bVBpeGVscyAqIG51bUNoYW5uZWxzKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG51bVBpeGVsczsgaSsrKSB7XG4gICAgICBmb3IgKGxldCBjaGFubmVsID0gMDsgY2hhbm5lbCA8IG51bUNoYW5uZWxzOyArK2NoYW5uZWwpIHtcbiAgICAgICAgdmFsdWVzW2kgKiBudW1DaGFubmVscyArIGNoYW5uZWxdID0gdmFsc1tpICogNCArIGNoYW5uZWxdO1xuICAgICAgfVxuICAgIH1cbiAgfVxuICBjb25zdCBvdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW2hlaWdodCwgd2lkdGgsIG51bUNoYW5uZWxzXTtcbiAgcmV0dXJuIHRlbnNvcjNkKHZhbHVlcywgb3V0U2hhcGUsICdpbnQzMicpO1xufVxuXG4vLyBIZWxwZXIgZnVuY3Rpb25zIGZvciB8ZnJvbVBpeGVsc0FzeW5jfCB0byBjaGVjayB3aGV0aGVyIHRoZSBpbnB1dCBjYW5cbi8vIGJlIHdyYXBwZWQgaW50byBpbWFnZUJpdG1hcC5cbmZ1bmN0aW9uIGlzUGl4ZWxEYXRhKHBpeGVsczogUGl4ZWxEYXRhfEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fFxuICAgICAgICAgICAgICAgICAgICAgSFRNTENhbnZhc0VsZW1lbnR8SFRNTFZpZGVvRWxlbWVudHxcbiAgICAgICAgICAgICAgICAgICAgIEltYWdlQml0bWFwKTogcGl4ZWxzIGlzIFBpeGVsRGF0YSB7XG4gIHJldHVybiAocGl4ZWxzICE9IG51bGwpICYmICgocGl4ZWxzIGFzIFBpeGVsRGF0YSkuZGF0YSBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkpO1xufVxuXG5mdW5jdGlvbiBpc0ltYWdlQml0bWFwRnVsbHlTdXBwb3J0ZWQoKSB7XG4gIHJldHVybiB0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICAgdHlwZW9mIChJbWFnZUJpdG1hcCkgIT09ICd1bmRlZmluZWQnICYmXG4gICAgICB3aW5kb3cuaGFzT3duUHJvcGVydHkoJ2NyZWF0ZUltYWdlQml0bWFwJyk7XG59XG5cbmZ1bmN0aW9uIGlzTm9uRW1wdHlQaXhlbHMocGl4ZWxzOiBQaXhlbERhdGF8SW1hZ2VEYXRhfEhUTUxJbWFnZUVsZW1lbnR8XG4gICAgICAgICAgICAgICAgICAgICAgICAgIEhUTUxDYW52YXNFbGVtZW50fEhUTUxWaWRlb0VsZW1lbnR8SW1hZ2VCaXRtYXApIHtcbiAgcmV0dXJuIHBpeGVscyAhPSBudWxsICYmIHBpeGVscy53aWR0aCAhPT0gMCAmJiBwaXhlbHMuaGVpZ2h0ICE9PSAwO1xufVxuXG5mdW5jdGlvbiBjYW5XcmFwUGl4ZWxzVG9JbWFnZUJpdG1hcChwaXhlbHM6IFBpeGVsRGF0YXxJbWFnZURhdGF8XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgSFRNTFZpZGVvRWxlbWVudHxJbWFnZUJpdG1hcCkge1xuICByZXR1cm4gaXNJbWFnZUJpdG1hcEZ1bGx5U3VwcG9ydGVkKCkgJiYgIShwaXhlbHMgaW5zdGFuY2VvZiBJbWFnZUJpdG1hcCkgJiZcbiAgICAgIGlzTm9uRW1wdHlQaXhlbHMocGl4ZWxzKSAmJiAhaXNQaXhlbERhdGEocGl4ZWxzKTtcbn1cblxuLyoqXG4gKiBDcmVhdGVzIGEgYHRmLlRlbnNvcmAgZnJvbSBhbiBpbWFnZSBpbiBhc3luYyB3YXkuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGltYWdlID0gbmV3IEltYWdlRGF0YSgxLCAxKTtcbiAqIGltYWdlLmRhdGFbMF0gPSAxMDA7XG4gKiBpbWFnZS5kYXRhWzFdID0gMTUwO1xuICogaW1hZ2UuZGF0YVsyXSA9IDIwMDtcbiAqIGltYWdlLmRhdGFbM10gPSAyNTU7XG4gKlxuICogKGF3YWl0IHRmLmJyb3dzZXIuZnJvbVBpeGVsc0FzeW5jKGltYWdlKSkucHJpbnQoKTtcbiAqIGBgYFxuICogVGhpcyBBUEkgaXMgdGhlIGFzeW5jIHZlcnNpb24gb2YgZnJvbVBpeGVscy4gVGhlIEFQSSB3aWxsIGZpcnN0XG4gKiBjaGVjayB8V1JBUF9UT19JTUFHRUJJVE1BUHwgZmxhZywgYW5kIHRyeSB0byB3cmFwIHRoZSBpbnB1dCB0b1xuICogaW1hZ2VCaXRtYXAgaWYgdGhlIGZsYWcgaXMgc2V0IHRvIHRydWUuXG4gKlxuICogQHBhcmFtIHBpeGVscyBUaGUgaW5wdXQgaW1hZ2UgdG8gY29uc3RydWN0IHRoZSB0ZW5zb3IgZnJvbS4gVGhlXG4gKiBzdXBwb3J0ZWQgaW1hZ2UgdHlwZXMgYXJlIGFsbCA0LWNoYW5uZWwuIFlvdSBjYW4gYWxzbyBwYXNzIGluIGFuIGltYWdlXG4gKiBvYmplY3Qgd2l0aCBmb2xsb3dpbmcgYXR0cmlidXRlczpcbiAqIGB7ZGF0YTogVWludDhBcnJheTsgd2lkdGg6IG51bWJlcjsgaGVpZ2h0OiBudW1iZXJ9YFxuICogQHBhcmFtIG51bUNoYW5uZWxzIFRoZSBudW1iZXIgb2YgY2hhbm5lbHMgb2YgdGhlIG91dHB1dCB0ZW5zb3IuIEFcbiAqIG51bUNoYW5uZWxzIHZhbHVlIGxlc3MgdGhhbiA0IGFsbG93cyB5b3UgdG8gaWdub3JlIGNoYW5uZWxzLiBEZWZhdWx0cyB0b1xuICogMyAoaWdub3JlcyBhbHBoYSBjaGFubmVsIG9mIGlucHV0IGltYWdlKS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnQnJvd3NlcicsIG5hbWVzcGFjZTogJ2Jyb3dzZXInLCBpZ25vcmVDSTogdHJ1ZX1cbiAqL1xuZXhwb3J0IGFzeW5jIGZ1bmN0aW9uIGZyb21QaXhlbHNBc3luYyhcbiAgICBwaXhlbHM6IFBpeGVsRGF0YXxJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxIVE1MQ2FudmFzRWxlbWVudHxcbiAgICBIVE1MVmlkZW9FbGVtZW50fEltYWdlQml0bWFwLFxuICAgIG51bUNoYW5uZWxzID0gMykge1xuICBsZXQgaW5wdXRzOiBQaXhlbERhdGF8SW1hZ2VEYXRhfEhUTUxJbWFnZUVsZW1lbnR8SFRNTENhbnZhc0VsZW1lbnR8XG4gICAgICBIVE1MVmlkZW9FbGVtZW50fEltYWdlQml0bWFwID0gbnVsbDtcblxuICAvLyBDaGVjayB3aGV0aGVyIHRoZSBiYWNrZW5kIG5lZWRzIHRvIHdyYXAgfHBpeGVsc3wgdG8gaW1hZ2VCaXRtYXAgYW5kXG4gIC8vIHdoZXRoZXIgfHBpeGVsc3wgY2FuIGJlIHdyYXBwZWQgdG8gaW1hZ2VCaXRtYXAuXG4gIGlmIChlbnYoKS5nZXRCb29sKCdXUkFQX1RPX0lNQUdFQklUTUFQJykgJiZcbiAgICAgIGNhbldyYXBQaXhlbHNUb0ltYWdlQml0bWFwKHBpeGVscykpIHtcbiAgICAvLyBGb3JjZSB0aGUgaW1hZ2VCaXRtYXAgY3JlYXRpb24gdG8gbm90IGRvIGFueSBwcmVtdWx0aXBseSBhbHBoYVxuICAgIC8vIG9wcy5cbiAgICBsZXQgaW1hZ2VCaXRtYXA7XG5cbiAgICB0cnkge1xuICAgICAgLy8gd3JhcCBpbiB0cnktY2F0Y2ggYmxvY2ssIGJlY2F1c2UgY3JlYXRlSW1hZ2VCaXRtYXAgbWF5IG5vdCB3b3JrXG4gICAgICAvLyBwcm9wZXJseSBpbiBzb21lIGJyb3dzZXJzLCBlLmcuXG4gICAgICAvLyBodHRwczovL2J1Z3ppbGxhLm1vemlsbGEub3JnL3Nob3dfYnVnLmNnaT9pZD0xMzM1NTk0XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWFueVxuICAgICAgaW1hZ2VCaXRtYXAgPSBhd2FpdCAoY3JlYXRlSW1hZ2VCaXRtYXAgYXMgYW55KShcbiAgICAgICAgICBwaXhlbHMgYXMgSW1hZ2VCaXRtYXBTb3VyY2UsIHtwcmVtdWx0aXBseUFscGhhOiAnbm9uZSd9KTtcbiAgICB9IGNhdGNoIChlKSB7XG4gICAgICBpbWFnZUJpdG1hcCA9IG51bGw7XG4gICAgfVxuXG4gICAgLy8gY3JlYXRlSW1hZ2VCaXRtYXAgd2lsbCBjbGlwIHRoZSBzb3VyY2Ugc2l6ZS5cbiAgICAvLyBJbiBzb21lIGNhc2VzLCB0aGUgaW5wdXQgd2lsbCBoYXZlIGxhcmdlciBzaXplIHRoYW4gaXRzIGNvbnRlbnQuXG4gICAgLy8gRS5nLiBuZXcgSW1hZ2UoMTAsIDEwKSBidXQgd2l0aCAxIHggMSBjb250ZW50LiBVc2luZ1xuICAgIC8vIGNyZWF0ZUltYWdlQml0bWFwIHdpbGwgY2xpcCB0aGUgc2l6ZSBmcm9tIDEwIHggMTAgdG8gMSB4IDEsIHdoaWNoXG4gICAgLy8gaXMgbm90IGNvcnJlY3QuIFdlIHNob3VsZCBhdm9pZCB3cmFwcGluZyBzdWNoIHJlc291Y2UgdG9cbiAgICAvLyBpbWFnZUJpdG1hcC5cbiAgICBpZiAoaW1hZ2VCaXRtYXAgIT0gbnVsbCAmJiBpbWFnZUJpdG1hcC53aWR0aCA9PT0gcGl4ZWxzLndpZHRoICYmXG4gICAgICAgIGltYWdlQml0bWFwLmhlaWdodCA9PT0gcGl4ZWxzLmhlaWdodCkge1xuICAgICAgaW5wdXRzID0gaW1hZ2VCaXRtYXA7XG4gICAgfSBlbHNlIHtcbiAgICAgIGlucHV0cyA9IHBpeGVscztcbiAgICB9XG4gIH0gZWxzZSB7XG4gICAgaW5wdXRzID0gcGl4ZWxzO1xuICB9XG5cbiAgcmV0dXJuIGZyb21QaXhlbHNfKGlucHV0cywgbnVtQ2hhbm5lbHMpO1xufVxuXG4vKipcbiAqIERyYXdzIGEgYHRmLlRlbnNvcmAgb2YgcGl4ZWwgdmFsdWVzIHRvIGEgYnl0ZSBhcnJheSBvciBvcHRpb25hbGx5IGFcbiAqIGNhbnZhcy5cbiAqXG4gKiBXaGVuIHRoZSBkdHlwZSBvZiB0aGUgaW5wdXQgaXMgJ2Zsb2F0MzInLCB3ZSBhc3N1bWUgdmFsdWVzIGluIHRoZSByYW5nZVxuICogWzAtMV0uIE90aGVyd2lzZSwgd2hlbiBpbnB1dCBpcyAnaW50MzInLCB3ZSBhc3N1bWUgdmFsdWVzIGluIHRoZSByYW5nZVxuICogWzAtMjU1XS5cbiAqXG4gKiBSZXR1cm5zIGEgcHJvbWlzZSB0aGF0IHJlc29sdmVzIHdoZW4gdGhlIGNhbnZhcyBoYXMgYmVlbiBkcmF3biB0by5cbiAqXG4gKiBAcGFyYW0gaW1nIEEgcmFuay0yIHRlbnNvciB3aXRoIHNoYXBlIGBbaGVpZ2h0LCB3aWR0aF1gLCBvciBhIHJhbmstMyB0ZW5zb3JcbiAqIG9mIHNoYXBlIGBbaGVpZ2h0LCB3aWR0aCwgbnVtQ2hhbm5lbHNdYC4gSWYgcmFuay0yLCBkcmF3cyBncmF5c2NhbGUuIElmXG4gKiByYW5rLTMsIG11c3QgaGF2ZSBkZXB0aCBvZiAxLCAzIG9yIDQuIFdoZW4gZGVwdGggb2YgMSwgZHJhd3NcbiAqIGdyYXlzY2FsZS4gV2hlbiBkZXB0aCBvZiAzLCB3ZSBkcmF3IHdpdGggdGhlIGZpcnN0IHRocmVlIGNvbXBvbmVudHMgb2ZcbiAqIHRoZSBkZXB0aCBkaW1lbnNpb24gY29ycmVzcG9uZGluZyB0byByLCBnLCBiIGFuZCBhbHBoYSA9IDEuIFdoZW4gZGVwdGggb2ZcbiAqIDQsIGFsbCBmb3VyIGNvbXBvbmVudHMgb2YgdGhlIGRlcHRoIGRpbWVuc2lvbiBjb3JyZXNwb25kIHRvIHIsIGcsIGIsIGEuXG4gKiBAcGFyYW0gY2FudmFzIFRoZSBjYW52YXMgdG8gZHJhdyB0by5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnQnJvd3NlcicsIG5hbWVzcGFjZTogJ2Jyb3dzZXInfVxuICovXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gdG9QaXhlbHMoXG4gICAgaW1nOiBUZW5zb3IyRHxUZW5zb3IzRHxUZW5zb3JMaWtlLFxuICAgIGNhbnZhcz86IEhUTUxDYW52YXNFbGVtZW50KTogUHJvbWlzZTxVaW50OENsYW1wZWRBcnJheT4ge1xuICBsZXQgJGltZyA9IGNvbnZlcnRUb1RlbnNvcihpbWcsICdpbWcnLCAndG9QaXhlbHMnKTtcbiAgaWYgKCEoaW1nIGluc3RhbmNlb2YgVGVuc29yKSkge1xuICAgIC8vIEFzc3VtZSBpbnQzMiBpZiB1c2VyIHBhc3NlZCBhIG5hdGl2ZSBhcnJheS5cbiAgICBjb25zdCBvcmlnaW5hbEltZ1RlbnNvciA9ICRpbWc7XG4gICAgJGltZyA9IGNhc3Qob3JpZ2luYWxJbWdUZW5zb3IsICdpbnQzMicpO1xuICAgIG9yaWdpbmFsSW1nVGVuc29yLmRpc3Bvc2UoKTtcbiAgfVxuICBpZiAoJGltZy5yYW5rICE9PSAyICYmICRpbWcucmFuayAhPT0gMykge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYHRvUGl4ZWxzIG9ubHkgc3VwcG9ydHMgcmFuayAyIG9yIDMgdGVuc29ycywgZ290IHJhbmsgJHskaW1nLnJhbmt9LmApO1xuICB9XG4gIGNvbnN0IFtoZWlnaHQsIHdpZHRoXSA9ICRpbWcuc2hhcGUuc2xpY2UoMCwgMik7XG4gIGNvbnN0IGRlcHRoID0gJGltZy5yYW5rID09PSAyID8gMSA6ICRpbWcuc2hhcGVbMl07XG5cbiAgaWYgKGRlcHRoID4gNCB8fCBkZXB0aCA9PT0gMikge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYHRvUGl4ZWxzIG9ubHkgc3VwcG9ydHMgZGVwdGggb2Ygc2l6ZSBgICtcbiAgICAgICAgYDEsIDMgb3IgNCBidXQgZ290ICR7ZGVwdGh9YCk7XG4gIH1cblxuICBpZiAoJGltZy5kdHlwZSAhPT0gJ2Zsb2F0MzInICYmICRpbWcuZHR5cGUgIT09ICdpbnQzMicpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBVbnN1cHBvcnRlZCB0eXBlIGZvciB0b1BpeGVsczogJHskaW1nLmR0eXBlfS5gICtcbiAgICAgICAgYCBQbGVhc2UgdXNlIGZsb2F0MzIgb3IgaW50MzIgdGVuc29ycy5gKTtcbiAgfVxuXG4gIGNvbnN0IGRhdGEgPSBhd2FpdCAkaW1nLmRhdGEoKTtcbiAgY29uc3QgbXVsdGlwbGllciA9ICRpbWcuZHR5cGUgPT09ICdmbG9hdDMyJyA/IDI1NSA6IDE7XG4gIGNvbnN0IGJ5dGVzID0gbmV3IFVpbnQ4Q2xhbXBlZEFycmF5KHdpZHRoICogaGVpZ2h0ICogNCk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBoZWlnaHQgKiB3aWR0aDsgKytpKSB7XG4gICAgY29uc3QgcmdiYSA9IFswLCAwLCAwLCAyNTVdO1xuXG4gICAgZm9yIChsZXQgZCA9IDA7IGQgPCBkZXB0aDsgZCsrKSB7XG4gICAgICBjb25zdCB2YWx1ZSA9IGRhdGFbaSAqIGRlcHRoICsgZF07XG5cbiAgICAgIGlmICgkaW1nLmR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICAgICAgaWYgKHZhbHVlIDwgMCB8fCB2YWx1ZSA+IDEpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAgIGBUZW5zb3IgdmFsdWVzIGZvciBhIGZsb2F0MzIgVGVuc29yIG11c3QgYmUgaW4gdGhlIGAgK1xuICAgICAgICAgICAgICBgcmFuZ2UgWzAgLSAxXSBidXQgZW5jb3VudGVyZWQgJHt2YWx1ZX0uYCk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAoJGltZy5kdHlwZSA9PT0gJ2ludDMyJykge1xuICAgICAgICBpZiAodmFsdWUgPCAwIHx8IHZhbHVlID4gMjU1KSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgICBgVGVuc29yIHZhbHVlcyBmb3IgYSBpbnQzMiBUZW5zb3IgbXVzdCBiZSBpbiB0aGUgYCArXG4gICAgICAgICAgICAgIGByYW5nZSBbMCAtIDI1NV0gYnV0IGVuY291bnRlcmVkICR7dmFsdWV9LmApO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChkZXB0aCA9PT0gMSkge1xuICAgICAgICByZ2JhWzBdID0gdmFsdWUgKiBtdWx0aXBsaWVyO1xuICAgICAgICByZ2JhWzFdID0gdmFsdWUgKiBtdWx0aXBsaWVyO1xuICAgICAgICByZ2JhWzJdID0gdmFsdWUgKiBtdWx0aXBsaWVyO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmdiYVtkXSA9IHZhbHVlICogbXVsdGlwbGllcjtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBjb25zdCBqID0gaSAqIDQ7XG4gICAgYnl0ZXNbaiArIDBdID0gTWF0aC5yb3VuZChyZ2JhWzBdKTtcbiAgICBieXRlc1tqICsgMV0gPSBNYXRoLnJvdW5kKHJnYmFbMV0pO1xuICAgIGJ5dGVzW2ogKyAyXSA9IE1hdGgucm91bmQocmdiYVsyXSk7XG4gICAgYnl0ZXNbaiArIDNdID0gTWF0aC5yb3VuZChyZ2JhWzNdKTtcbiAgfVxuXG4gIGlmIChjYW52YXMgIT0gbnVsbCkge1xuICAgIGNhbnZhcy53aWR0aCA9IHdpZHRoO1xuICAgIGNhbnZhcy5oZWlnaHQgPSBoZWlnaHQ7XG4gICAgY29uc3QgY3R4ID0gY2FudmFzLmdldENvbnRleHQoJzJkJyk7XG4gICAgY29uc3QgaW1hZ2VEYXRhID0gbmV3IEltYWdlRGF0YShieXRlcywgd2lkdGgsIGhlaWdodCk7XG4gICAgY3R4LnB1dEltYWdlRGF0YShpbWFnZURhdGEsIDAsIDApO1xuICB9XG4gIGlmICgkaW1nICE9PSBpbWcpIHtcbiAgICAkaW1nLmRpc3Bvc2UoKTtcbiAgfVxuICByZXR1cm4gYnl0ZXM7XG59XG5cbmV4cG9ydCBjb25zdCBmcm9tUGl4ZWxzID0gLyogQF9fUFVSRV9fICovIG9wKHtmcm9tUGl4ZWxzX30pO1xuIl19