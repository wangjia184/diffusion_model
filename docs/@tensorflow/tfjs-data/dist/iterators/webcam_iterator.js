/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
 *
 * =============================================================================
 */
import { browser, cast, env, expandDims, image, reshape, tensor1d, tensor2d, tidy, util } from '@tensorflow/tfjs-core';
import { LazyIterator } from './lazy_iterator';
/**
 * Provide a stream of image tensors from webcam video stream. Only works in
 * browser environment.
 */
export class WebcamIterator extends LazyIterator {
    constructor(webcamVideoElement, webcamConfig) {
        super();
        this.webcamVideoElement = webcamVideoElement;
        this.webcamConfig = webcamConfig;
        this.isClosed = true;
        this.resize = false;
        if (this.needToResize()) {
            this.resize = true;
            this.cropSize =
                [this.webcamConfig.resizeHeight, this.webcamConfig.resizeWidth];
            this.cropBoxInd = tensor1d([0], 'int32');
            if (this.webcamConfig.centerCrop) {
                // Calculate the box based on resizing shape.
                const widthCroppingRatio = this.webcamConfig.resizeWidth * 1.0 / this.webcamVideoElement.width;
                const heightCroppingRatio = this.webcamConfig.resizeHeight * 1.0 /
                    this.webcamVideoElement.height;
                const widthCropStart = (1 - widthCroppingRatio) / 2;
                const heightCropStart = (1 - heightCroppingRatio) / 2;
                const widthCropEnd = widthCropStart + widthCroppingRatio;
                const heightCropEnd = heightCroppingRatio + heightCropStart;
                this.cropBox = tensor2d([heightCropStart, widthCropStart, heightCropEnd, widthCropEnd], [1, 4]);
            }
            else {
                this.cropBox = tensor2d([0, 0, 1, 1], [1, 4]);
            }
        }
    }
    summary() {
        return `webcam`;
    }
    // Construct a WebcamIterator and start it's video stream.
    static async create(webcamVideoElement, webcamConfig = {}) {
        if (!env().get('IS_BROWSER')) {
            throw new Error('tf.data.webcam is only supported in browser environment.');
        }
        if (!webcamVideoElement) {
            // If webcam video element is not provided, create a hidden video element
            // with provided width and height.
            webcamVideoElement = document.createElement('video');
            if (!webcamConfig.resizeWidth || !webcamConfig.resizeHeight) {
                throw new Error('Please provide webcam video element, or resizeWidth and ' +
                    'resizeHeight to create a hidden video element.');
            }
            webcamVideoElement.width = webcamConfig.resizeWidth;
            webcamVideoElement.height = webcamConfig.resizeHeight;
        }
        const webcamIterator = new WebcamIterator(webcamVideoElement, webcamConfig);
        // Call async function to initialize the video stream.
        await webcamIterator.start();
        return webcamIterator;
    }
    // Async function to start video stream.
    async start() {
        if (this.webcamConfig.facingMode) {
            util.assert((this.webcamConfig.facingMode === 'user') ||
                (this.webcamConfig.facingMode === 'environment'), () => `Invalid webcam facing mode: ${this.webcamConfig.facingMode}. ` +
                `Please provide 'user' or 'environment'`);
        }
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    deviceId: this.webcamConfig.deviceId,
                    facingMode: this.webcamConfig.facingMode ?
                        this.webcamConfig.facingMode :
                        'user',
                    width: this.webcamVideoElement.width,
                    height: this.webcamVideoElement.height
                }
            });
        }
        catch (e) {
            // Modify the error message but leave the stack trace intact
            e.message = `Error thrown while initializing video stream: ${e.message}`;
            throw e;
        }
        if (!this.stream) {
            throw new Error('Could not obtain video from webcam.');
        }
        // Older browsers may not have srcObject
        try {
            this.webcamVideoElement.srcObject = this.stream;
        }
        catch (error) {
            console.log(error);
            this.webcamVideoElement.src = window.URL.createObjectURL(this.stream);
        }
        // Start the webcam video stream
        this.webcamVideoElement.play();
        this.isClosed = false;
        return new Promise(resolve => {
            // Add event listener to make sure the webcam has been fully initialized.
            this.webcamVideoElement.onloadedmetadata = () => {
                resolve();
            };
        });
    }
    async next() {
        if (this.isClosed) {
            return { value: null, done: true };
        }
        let img;
        try {
            img = browser.fromPixels(this.webcamVideoElement);
        }
        catch (e) {
            throw new Error(`Error thrown converting video to pixels: ${JSON.stringify(e)}`);
        }
        if (this.resize) {
            try {
                return { value: this.cropAndResizeFrame(img), done: false };
            }
            catch (e) {
                throw new Error(`Error thrown cropping the video: ${e.message}`);
            }
            finally {
                img.dispose();
            }
        }
        else {
            return { value: img, done: false };
        }
    }
    needToResize() {
        // If resizeWidth and resizeHeight are provided, and different from the
        // width and height of original HTMLVideoElement, then resizing and cropping
        // is required.
        if (this.webcamConfig.resizeWidth && this.webcamConfig.resizeHeight &&
            (this.webcamVideoElement.width !== this.webcamConfig.resizeWidth ||
                this.webcamVideoElement.height !== this.webcamConfig.resizeHeight)) {
            return true;
        }
        return false;
    }
    // Cropping and resizing each frame based on config
    cropAndResizeFrame(img) {
        return tidy(() => {
            const expandedImage = expandDims(cast(img, 'float32'), (0));
            let resizedImage;
            resizedImage = image.cropAndResize(expandedImage, this.cropBox, this.cropBoxInd, this.cropSize, 'bilinear');
            // Extract image from batch cropping.
            const shape = resizedImage.shape;
            return reshape(resizedImage, shape.slice(1));
        });
    }
    // Capture one frame from the video stream, and extract the value from
    // iterator.next() result.
    async capture() {
        return (await this.next()).value;
    }
    // Stop the video stream and pause webcam iterator.
    stop() {
        const tracks = this.stream.getTracks();
        tracks.forEach(track => track.stop());
        try {
            this.webcamVideoElement.srcObject = null;
        }
        catch (error) {
            console.log(error);
            this.webcamVideoElement.src = null;
        }
        this.isClosed = true;
    }
    // Override toArray() function to prevent collecting.
    toArray() {
        throw new Error('Can not convert infinite video stream to array.');
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid2ViY2FtX2l0ZXJhdG9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1kYXRhL3NyYy9pdGVyYXRvcnMvd2ViY2FtX2l0ZXJhdG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7O0dBZ0JHO0FBRUgsT0FBTyxFQUFDLE9BQU8sRUFBRSxJQUFJLEVBQUUsR0FBRyxFQUFFLFVBQVUsRUFBRSxLQUFLLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBWSxRQUFRLEVBQWdDLElBQUksRUFBRSxJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUU3SixPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFFN0M7OztHQUdHO0FBQ0gsTUFBTSxPQUFPLGNBQWUsU0FBUSxZQUFzQjtJQVF4RCxZQUN1QixrQkFBb0MsRUFDcEMsWUFBMEI7UUFDL0MsS0FBSyxFQUFFLENBQUM7UUFGYSx1QkFBa0IsR0FBbEIsa0JBQWtCLENBQWtCO1FBQ3BDLGlCQUFZLEdBQVosWUFBWSxDQUFjO1FBVHpDLGFBQVEsR0FBRyxJQUFJLENBQUM7UUFFaEIsV0FBTSxHQUFHLEtBQUssQ0FBQztRQVNyQixJQUFJLElBQUksQ0FBQyxZQUFZLEVBQUUsRUFBRTtZQUN2QixJQUFJLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztZQUNuQixJQUFJLENBQUMsUUFBUTtnQkFDVCxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsWUFBWSxFQUFFLElBQUksQ0FBQyxZQUFZLENBQUMsV0FBVyxDQUFDLENBQUM7WUFDcEUsSUFBSSxDQUFDLFVBQVUsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsQ0FBQztZQUN6QyxJQUFJLElBQUksQ0FBQyxZQUFZLENBQUMsVUFBVSxFQUFFO2dCQUNoQyw2Q0FBNkM7Z0JBQzdDLE1BQU0sa0JBQWtCLEdBQ3BCLElBQUksQ0FBQyxZQUFZLENB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Q0FBQyxDQUFDO1lBQ25CLElBQUksQ0FBQyxrQkFBa0IsQ0FBQyxHQUFHLEdBQUcsSUFBSSxDQUFDO1NBQ3BDO1FBQ0QsSUFBSSxDQUFDLFFBQVEsR0FBRyxJQUFJLENBQUM7SUFDdkIsQ0FBQztJQUVELHFEQUFxRDtJQUM1QyxPQUFPO1FBQ2QsTUFBTSxJQUFJLEtBQUssQ0FBQyxpREFBaUQsQ0FBQyxDQUFDO0lBQ3JFLENBQUM7Q0FDRiIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YnJvd3NlciwgY2FzdCwgZW52LCBleHBhbmREaW1zLCBpbWFnZSwgcmVzaGFwZSwgdGVuc29yMWQsIFRlbnNvcjFELCB0ZW5zb3IyZCwgVGVuc29yMkQsIFRlbnNvcjNELCBUZW5zb3I0RCwgdGlkeSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7V2ViY2FtQ29uZmlnfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge0xhenlJdGVyYXRvcn0gZnJvbSAnLi9sYXp5X2l0ZXJhdG9yJztcblxuLyoqXG4gKiBQcm92aWRlIGEgc3RyZWFtIG9mIGltYWdlIHRlbnNvcnMgZnJvbSB3ZWJjYW0gdmlkZW8gc3RyZWFtLiBPbmx5IHdvcmtzIGluXG4gKiBicm93c2VyIGVudmlyb25tZW50LlxuICovXG5leHBvcnQgY2xhc3MgV2ViY2FtSXRlcmF0b3IgZXh0ZW5kcyBMYXp5SXRlcmF0b3I8VGVuc29yM0Q+IHtcbiAgcHJpdmF0ZSBpc0Nsb3NlZCA9IHRydWU7XG4gIHByaXZhdGUgc3RyZWFtOiBNZWRpYVN0cmVhbTtcbiAgcHJpdmF0ZSByZXNpemUgPSBmYWxzZTtcbiAgcHJpdmF0ZSBjcm9wU2l6ZTogW251bWJlciwgbnVtYmVyXTtcbiAgcHJpdmF0ZSBjcm9wQm94OiBUZW5zb3IyRDtcbiAgcHJpdmF0ZSBjcm9wQm94SW5kOiBUZW5zb3IxRDtcblxuICBwcml2YXRlIGNvbnN0cnVjdG9yKFxuICAgICAgcHJvdGVjdGVkIHJlYWRvbmx5IHdlYmNhbVZpZGVvRWxlbWVudDogSFRNTFZpZGVvRWxlbWVudCxcbiAgICAgIHByb3RlY3RlZCByZWFkb25seSB3ZWJjYW1Db25maWc6IFdlYmNhbUNvbmZpZykge1xuICAgIHN1cGVyKCk7XG4gICAgaWYgKHRoaXMubmVlZFRvUmVzaXplKCkpIHtcbiAgICAgIHRoaXMucmVzaXplID0gdHJ1ZTtcbiAgICAgIHRoaXMuY3JvcFNpemUgPVxuICAgICAgICAgIFt0aGlzLndlYmNhbUNvbmZpZy5yZXNpemVIZWlnaHQsIHRoaXMud2ViY2FtQ29uZmlnLnJlc2l6ZVdpZHRoXTtcbiAgICAgIHRoaXMuY3JvcEJveEluZCA9IHRlbnNvcjFkKFswXSwgJ2ludDMyJyk7XG4gICAgICBpZiAodGhpcy53ZWJjYW1Db25maWcuY2VudGVyQ3JvcCkge1xuICAgICAgICAvLyBDYWxjdWxhdGUgdGhlIGJveCBiYXNlZCBvbiByZXNpemluZyBzaGFwZS5cbiAgICAgICAgY29uc3Qgd2lkdGhDcm9wcGluZ1JhdGlvID1cbiAgICAgICAgICAgIHRoaXMud2ViY2FtQ29uZmlnLnJlc2l6ZVdpZHRoICogMS4wIC8gdGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQud2lkdGg7XG4gICAgICAgIGNvbnN0IGhlaWdodENyb3BwaW5nUmF0aW8gPSB0aGlzLndlYmNhbUNvbmZpZy5yZXNpemVIZWlnaHQgKiAxLjAgL1xuICAgICAgICAgICAgdGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQuaGVpZ2h0O1xuICAgICAgICBjb25zdCB3aWR0aENyb3BTdGFydCA9ICgxIC0gd2lkdGhDcm9wcGluZ1JhdGlvKSAvIDI7XG4gICAgICAgIGNvbnN0IGhlaWdodENyb3BTdGFydCA9ICgxIC0gaGVpZ2h0Q3JvcHBpbmdSYXRpbykgLyAyO1xuICAgICAgICBjb25zdCB3aWR0aENyb3BFbmQgPSB3aWR0aENyb3BTdGFydCArIHdpZHRoQ3JvcHBpbmdSYXRpbztcbiAgICAgICAgY29uc3QgaGVpZ2h0Q3JvcEVuZCA9IGhlaWdodENyb3BwaW5nUmF0aW8gKyBoZWlnaHRDcm9wU3RhcnQ7XG4gICAgICAgIHRoaXMuY3JvcEJveCA9IHRlbnNvcjJkKFxuICAgICAgICAgICAgW2hlaWdodENyb3BTdGFydCwgd2lkdGhDcm9wU3RhcnQsIGhlaWdodENyb3BFbmQsIHdpZHRoQ3JvcEVuZF0sXG4gICAgICAgICAgICBbMSwgNF0pO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhpcy5jcm9wQm94ID0gdGVuc29yMmQoWzAsIDAsIDEsIDFdLCBbMSwgNF0pO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIHN1bW1hcnkoKSB7XG4gICAgcmV0dXJuIGB3ZWJjYW1gO1xuICB9XG5cbiAgLy8gQ29uc3RydWN0IGEgV2ViY2FtSXRlcmF0b3IgYW5kIHN0YXJ0IGl0J3MgdmlkZW8gc3RyZWFtLlxuICBzdGF0aWMgYXN5bmMgY3JlYXRlKFxuICAgICAgd2ViY2FtVmlkZW9FbGVtZW50PzogSFRNTFZpZGVvRWxlbWVudCwgd2ViY2FtQ29uZmlnOiBXZWJjYW1Db25maWcgPSB7fSkge1xuICAgIGlmICghZW52KCkuZ2V0KCdJU19CUk9XU0VSJykpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAndGYuZGF0YS53ZWJjYW0gaXMgb25seSBzdXBwb3J0ZWQgaW4gYnJvd3NlciBlbnZpcm9ubWVudC4nKTtcbiAgICB9XG5cbiAgICBpZiAoIXdlYmNhbVZpZGVvRWxlbWVudCkge1xuICAgICAgLy8gSWYgd2ViY2FtIHZpZGVvIGVsZW1lbnQgaXMgbm90IHByb3ZpZGVkLCBjcmVhdGUgYSBoaWRkZW4gdmlkZW8gZWxlbWVudFxuICAgICAgLy8gd2l0aCBwcm92aWRlZCB3aWR0aCBhbmQgaGVpZ2h0LlxuICAgICAgd2ViY2FtVmlkZW9FbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgndmlkZW8nKTtcbiAgICAgIGlmICghd2ViY2FtQ29uZmlnLnJlc2l6ZVdpZHRoIHx8ICF3ZWJjYW1Db25maWcucmVzaXplSGVpZ2h0KSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdQbGVhc2UgcHJvdmlkZSB3ZWJjYW0gdmlkZW8gZWxlbWVudCwgb3IgcmVzaXplV2lkdGggYW5kICcgK1xuICAgICAgICAgICAgJ3Jlc2l6ZUhlaWdodCB0byBjcmVhdGUgYSBoaWRkZW4gdmlkZW8gZWxlbWVudC4nKTtcbiAgICAgIH1cbiAgICAgIHdlYmNhbVZpZGVvRWxlbWVudC53aWR0aCA9IHdlYmNhbUNvbmZpZy5yZXNpemVXaWR0aDtcbiAgICAgIHdlYmNhbVZpZGVvRWxlbWVudC5oZWlnaHQgPSB3ZWJjYW1Db25maWcucmVzaXplSGVpZ2h0O1xuICAgIH1cbiAgICBjb25zdCB3ZWJjYW1JdGVyYXRvciA9IG5ldyBXZWJjYW1JdGVyYXRvcih3ZWJjYW1WaWRlb0VsZW1lbnQsIHdlYmNhbUNvbmZpZyk7XG5cbiAgICAvLyBDYWxsIGFzeW5jIGZ1bmN0aW9uIHRvIGluaXRpYWxpemUgdGhlIHZpZGVvIHN0cmVhbS5cbiAgICBhd2FpdCB3ZWJjYW1JdGVyYXRvci5zdGFydCgpO1xuXG4gICAgcmV0dXJuIHdlYmNhbUl0ZXJhdG9yO1xuICB9XG5cbiAgLy8gQXN5bmMgZnVuY3Rpb24gdG8gc3RhcnQgdmlkZW8gc3RyZWFtLlxuICBhc3luYyBzdGFydCgpOiBQcm9taXNlPHZvaWQ+IHtcbiAgICBpZiAodGhpcy53ZWJjYW1Db25maWcuZmFjaW5nTW9kZSkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgKHRoaXMud2ViY2FtQ29uZmlnLmZhY2luZ01vZGUgPT09ICd1c2VyJykgfHxcbiAgICAgICAgICAgICAgKHRoaXMud2ViY2FtQ29uZmlnLmZhY2luZ01vZGUgPT09ICdlbnZpcm9ubWVudCcpLFxuICAgICAgICAgICgpID0+XG4gICAgICAgICAgICAgIGBJbnZhbGlkIHdlYmNhbSBmYWNpbmcgbW9kZTogJHt0aGlzLndlYmNhbUNvbmZpZy5mYWNpbmdNb2RlfS4gYCArXG4gICAgICAgICAgICAgIGBQbGVhc2UgcHJvdmlkZSAndXNlcicgb3IgJ2Vudmlyb25tZW50J2ApO1xuICAgIH1cblxuICAgIHRyeSB7XG4gICAgICB0aGlzLnN0cmVhbSA9IGF3YWl0IG5hdmlnYXRvci5tZWRpYURldmljZXMuZ2V0VXNlck1lZGlhKHtcbiAgICAgICAgdmlkZW86IHtcbiAgICAgICAgICBkZXZpY2VJZDogdGhpcy53ZWJjYW1Db25maWcuZGV2aWNlSWQsXG4gICAgICAgICAgZmFjaW5nTW9kZTogdGhpcy53ZWJjYW1Db25maWcuZmFjaW5nTW9kZSA/XG4gICAgICAgICAgICAgIHRoaXMud2ViY2FtQ29uZmlnLmZhY2luZ01vZGUgOlxuICAgICAgICAgICAgICAndXNlcicsXG4gICAgICAgICAgd2lkdGg6IHRoaXMud2ViY2FtVmlkZW9FbGVtZW50LndpZHRoLFxuICAgICAgICAgIGhlaWdodDogdGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQuaGVpZ2h0XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgIC8vIE1vZGlmeSB0aGUgZXJyb3IgbWVzc2FnZSBidXQgbGVhdmUgdGhlIHN0YWNrIHRyYWNlIGludGFjdFxuICAgICAgZS5tZXNzYWdlID0gYEVycm9yIHRocm93biB3aGlsZSBpbml0aWFsaXppbmcgdmlkZW8gc3RyZWFtOiAke2UubWVzc2FnZX1gO1xuICAgICAgdGhyb3cgZTtcbiAgICB9XG5cbiAgICBpZiAoIXRoaXMuc3RyZWFtKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0NvdWxkIG5vdCBvYnRhaW4gdmlkZW8gZnJvbSB3ZWJjYW0uJyk7XG4gICAgfVxuXG4gICAgLy8gT2xkZXIgYnJvd3NlcnMgbWF5IG5vdCBoYXZlIHNyY09iamVjdFxuICAgIHRyeSB7XG4gICAgICB0aGlzLndlYmNhbVZpZGVvRWxlbWVudC5zcmNPYmplY3QgPSB0aGlzLnN0cmVhbTtcbiAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgY29uc29sZS5sb2coZXJyb3IpO1xuICAgICAgdGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQuc3JjID0gd2luZG93LlVSTC5jcmVhdGVPYmplY3RVUkwoXG4gICAgICAgIHRoaXMuc3RyZWFtIGFzIHVua25vd24gYXMgTWVkaWFTb3VyY2UpO1xuICAgIH1cbiAgICAvLyBTdGFydCB0aGUgd2ViY2FtIHZpZGVvIHN0cmVhbVxuICAgIHRoaXMud2ViY2FtVmlkZW9FbGVtZW50LnBsYXkoKTtcblxuICAgIHRoaXMuaXNDbG9zZWQgPSBmYWxzZTtcblxuICAgIHJldHVybiBuZXcgUHJvbWlzZTx2b2lkPihyZXNvbHZlID0+IHtcbiAgICAgIC8vIEFkZCBldmVudCBsaXN0ZW5lciB0byBtYWtlIHN1cmUgdGhlIHdlYmNhbSBoYXMgYmVlbiBmdWxseSBpbml0aWFsaXplZC5cbiAgICAgIHRoaXMud2ViY2FtVmlkZW9FbGVtZW50Lm9ubG9hZGVkbWV0YWRhdGEgPSAoKSA9PiB7XG4gICAgICAgIHJlc29sdmUoKTtcbiAgICAgIH07XG4gICAgfSk7XG4gIH1cblxuICBhc3luYyBuZXh0KCk6IFByb21pc2U8SXRlcmF0b3JSZXN1bHQ8VGVuc29yM0Q+PiB7XG4gICAgaWYgKHRoaXMuaXNDbG9zZWQpIHtcbiAgICAgIHJldHVybiB7dmFsdWU6IG51bGwsIGRvbmU6IHRydWV9O1xuICAgIH1cblxuICAgIGxldCBpbWc7XG4gICAgdHJ5IHtcbiAgICAgIGltZyA9IGJyb3dzZXIuZnJvbVBpeGVscyh0aGlzLndlYmNhbVZpZGVvRWxlbWVudCk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBFcnJvciB0aHJvd24gY29udmVydGluZyB2aWRlbyB0byBwaXhlbHM6ICR7SlNPTi5zdHJpbmdpZnkoZSl9YCk7XG4gICAgfVxuICAgIGlmICh0aGlzLnJlc2l6ZSkge1xuICAgICAgdHJ5IHtcbiAgICAgICAgcmV0dXJuIHt2YWx1ZTogdGhpcy5jcm9wQW5kUmVzaXplRnJhbWUoaW1nKSwgZG9uZTogZmFsc2V9O1xuICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYEVycm9yIHRocm93biBjcm9wcGluZyB0aGUgdmlkZW86ICR7ZS5tZXNzYWdlfWApO1xuICAgICAgfSBmaW5hbGx5IHtcbiAgICAgICAgaW1nLmRpc3Bvc2UoKTtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgcmV0dXJuIHt2YWx1ZTogaW1nLCBkb25lOiBmYWxzZX07XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBuZWVkVG9SZXNpemUoKSB7XG4gICAgLy8gSWYgcmVzaXplV2lkdGggYW5kIHJlc2l6ZUhlaWdodCBhcmUgcHJvdmlkZWQsIGFuZCBkaWZmZXJlbnQgZnJvbSB0aGVcbiAgICAvLyB3aWR0aCBhbmQgaGVpZ2h0IG9mIG9yaWdpbmFsIEhUTUxWaWRlb0VsZW1lbnQsIHRoZW4gcmVzaXppbmcgYW5kIGNyb3BwaW5nXG4gICAgLy8gaXMgcmVxdWlyZWQuXG4gICAgaWYgKHRoaXMud2ViY2FtQ29uZmlnLnJlc2l6ZVdpZHRoICYmIHRoaXMud2ViY2FtQ29uZmlnLnJlc2l6ZUhlaWdodCAmJlxuICAgICAgICAodGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQud2lkdGggIT09IHRoaXMud2ViY2FtQ29uZmlnLnJlc2l6ZVdpZHRoIHx8XG4gICAgICAgICB0aGlzLndlYmNhbVZpZGVvRWxlbWVudC5oZWlnaHQgIT09IHRoaXMud2ViY2FtQ29uZmlnLnJlc2l6ZUhlaWdodCkpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICAvLyBDcm9wcGluZyBhbmQgcmVzaXppbmcgZWFjaCBmcmFtZSBiYXNlZCBvbiBjb25maWdcbiAgY3JvcEFuZFJlc2l6ZUZyYW1lKGltZzogVGVuc29yM0QpOiBUZW5zb3IzRCB7XG4gICAgcmV0dXJuIHRpZHkoKCkgPT4ge1xuICAgICAgY29uc3QgZXhwYW5kZWRJbWFnZTogVGVuc29yNEQgPSBleHBhbmREaW1zKGNhc3QoaW1nLCAnZmxvYXQzMicpLCAoMCkpO1xuICAgICAgbGV0IHJlc2l6ZWRJbWFnZTtcbiAgICAgIHJlc2l6ZWRJbWFnZSA9IGltYWdlLmNyb3BBbmRSZXNpemUoXG4gICAgICAgICAgZXhwYW5kZWRJbWFnZSwgdGhpcy5jcm9wQm94LCB0aGlzLmNyb3BCb3hJbmQsIHRoaXMuY3JvcFNpemUsXG4gICAgICAgICAgJ2JpbGluZWFyJyk7XG4gICAgICAvLyBFeHRyYWN0IGltYWdlIGZyb20gYmF0Y2ggY3JvcHBpbmcuXG4gICAgICBjb25zdCBzaGFwZSA9IHJlc2l6ZWRJbWFnZS5zaGFwZTtcbiAgICAgIHJldHVybiByZXNoYXBlKHJlc2l6ZWRJbWFnZSwgc2hhcGUuc2xpY2UoMSkgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTtcbiAgICB9KTtcbiAgfVxuXG4gIC8vIENhcHR1cmUgb25lIGZyYW1lIGZyb20gdGhlIHZpZGVvIHN0cmVhbSwgYW5kIGV4dHJhY3QgdGhlIHZhbHVlIGZyb21cbiAgLy8gaXRlcmF0b3IubmV4dCgpIHJlc3VsdC5cbiAgYXN5bmMgY2FwdHVyZSgpOiBQcm9taXNlPFRlbnNvcjNEPiB7XG4gICAgcmV0dXJuIChhd2FpdCB0aGlzLm5leHQoKSkudmFsdWU7XG4gIH1cblxuICAvLyBTdG9wIHRoZSB2aWRlbyBzdHJlYW0gYW5kIHBhdXNlIHdlYmNhbSBpdGVyYXRvci5cbiAgc3RvcCgpOiB2b2lkIHtcbiAgICBjb25zdCB0cmFja3MgPSB0aGlzLnN0cmVhbS5nZXRUcmFja3MoKTtcblxuICAgIHRyYWNrcy5mb3JFYWNoKHRyYWNrID0+IHRyYWNrLnN0b3AoKSk7XG5cbiAgICB0cnkge1xuICAgICAgdGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQuc3JjT2JqZWN0ID0gbnVsbDtcbiAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgY29uc29sZS5sb2coZXJyb3IpO1xuICAgICAgdGhpcy53ZWJjYW1WaWRlb0VsZW1lbnQuc3JjID0gbnVsbDtcbiAgICB9XG4gICAgdGhpcy5pc0Nsb3NlZCA9IHRydWU7XG4gIH1cblxuICAvLyBPdmVycmlkZSB0b0FycmF5KCkgZnVuY3Rpb24gdG8gcHJldmVudCBjb2xsZWN0aW5nLlxuICBvdmVycmlkZSB0b0FycmF5KCk6IFByb21pc2U8VGVuc29yM0RbXT4ge1xuICAgIHRocm93IG5ldyBFcnJvcignQ2FuIG5vdCBjb252ZXJ0IGluZmluaXRlIHZpZGVvIHN0cmVhbSB0byBhcnJheS4nKTtcbiAgfVxufVxuIl19