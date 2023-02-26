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
import { mul } from '../mul';
import { op } from '../operation';
import { enclosingPowerOfTwo } from '../signal_ops_util';
import { rfft } from '../spectral/rfft';
import { frame } from './frame';
import { hannWindow } from './hann_window';
/**
 * Computes the Short-time Fourier Transform of signals
 * See: https://en.wikipedia.org/wiki/Short-time_Fourier_transform
 *
 * ```js
 * const input = tf.tensor1d([1, 1, 1, 1, 1])
 * tf.signal.stft(input, 3, 1).print();
 * ```
 * @param signal 1-dimensional real value tensor.
 * @param frameLength The window length of samples.
 * @param frameStep The number of samples to step.
 * @param fftLength The size of the FFT to apply.
 * @param windowFn A callable that takes a window length and returns 1-d tensor.
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function stft_(signal, frameLength, frameStep, fftLength, windowFn = hannWindow) {
    if (fftLength == null) {
        fftLength = enclosingPowerOfTwo(frameLength);
    }
    const framedSignal = frame(signal, frameLength, frameStep);
    const windowedSignal = mul(framedSignal, windowFn(frameLength));
    return rfft(windowedSignal, fftLength);
}
export const stft = /* @__PURE__ */ op({ stft_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3RmdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NpZ25hbC9zdGZ0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUN2RCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFFdEMsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUM5QixPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBRXpDOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILFNBQVMsS0FBSyxDQUNWLE1BQWdCLEVBQUUsV0FBbUIsRUFBRSxTQUFpQixFQUN4RCxTQUFrQixFQUNsQixXQUF5QyxVQUFVO0lBQ3JELElBQUksU0FBUyxJQUFJLElBQUksRUFBRTtRQUNyQixTQUFTLEdBQUcsbUJBQW1CLENBQUMsV0FBVyxDQUFDLENBQUM7S0FDOUM7SUFDRCxNQUFNLFlBQVksR0FBRyxLQUFLLENBQUMsTUFBTSxFQUFFLFdBQVcsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUMzRCxNQUFNLGNBQWMsR0FBRyxHQUFHLENBQUMsWUFBWSxFQUFFLFFBQVEsQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLE9BQU8sSUFBSSxDQUFDLGNBQWMsRUFBRSxTQUFTLENBQUMsQ0FBQztBQUN6QyxDQUFDO0FBQ0QsTUFBTSxDQUFDLE1BQU0sSUFBSSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxLQUFLLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvciwgVGVuc29yMUR9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge211bH0gZnJvbSAnLi4vbXVsJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge2VuY2xvc2luZ1Bvd2VyT2ZUd299IGZyb20gJy4uL3NpZ25hbF9vcHNfdXRpbCc7XG5pbXBvcnQge3JmZnR9IGZyb20gJy4uL3NwZWN0cmFsL3JmZnQnO1xuXG5pbXBvcnQge2ZyYW1lfSBmcm9tICcuL2ZyYW1lJztcbmltcG9ydCB7aGFubldpbmRvd30gZnJvbSAnLi9oYW5uX3dpbmRvdyc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIFNob3J0LXRpbWUgRm91cmllciBUcmFuc2Zvcm0gb2Ygc2lnbmFsc1xuICogU2VlOiBodHRwczovL2VuLndpa2lwZWRpYS5vcmcvd2lraS9TaG9ydC10aW1lX0ZvdXJpZXJfdHJhbnNmb3JtXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGlucHV0ID0gdGYudGVuc29yMWQoWzEsIDEsIDEsIDEsIDFdKVxuICogdGYuc2lnbmFsLnN0ZnQoaW5wdXQsIDMsIDEpLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSBzaWduYWwgMS1kaW1lbnNpb25hbCByZWFsIHZhbHVlIHRlbnNvci5cbiAqIEBwYXJhbSBmcmFtZUxlbmd0aCBUaGUgd2luZG93IGxlbmd0aCBvZiBzYW1wbGVzLlxuICogQHBhcmFtIGZyYW1lU3RlcCBUaGUgbnVtYmVyIG9mIHNhbXBsZXMgdG8gc3RlcC5cbiAqIEBwYXJhbSBmZnRMZW5ndGggVGhlIHNpemUgb2YgdGhlIEZGVCB0byBhcHBseS5cbiAqIEBwYXJhbSB3aW5kb3dGbiBBIGNhbGxhYmxlIHRoYXQgdGFrZXMgYSB3aW5kb3cgbGVuZ3RoIGFuZCByZXR1cm5zIDEtZCB0ZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU2lnbmFsJywgbmFtZXNwYWNlOiAnc2lnbmFsJ31cbiAqL1xuZnVuY3Rpb24gc3RmdF8oXG4gICAgc2lnbmFsOiBUZW5zb3IxRCwgZnJhbWVMZW5ndGg6IG51bWJlciwgZnJhbWVTdGVwOiBudW1iZXIsXG4gICAgZmZ0TGVuZ3RoPzogbnVtYmVyLFxuICAgIHdpbmRvd0ZuOiAobGVuZ3RoOiBudW1iZXIpID0+IFRlbnNvcjFEID0gaGFubldpbmRvdyk6IFRlbnNvciB7XG4gIGlmIChmZnRMZW5ndGggPT0gbnVsbCkge1xuICAgIGZmdExlbmd0aCA9IGVuY2xvc2luZ1Bvd2VyT2ZUd28oZnJhbWVMZW5ndGgpO1xuICB9XG4gIGNvbnN0IGZyYW1lZFNpZ25hbCA9IGZyYW1lKHNpZ25hbCwgZnJhbWVMZW5ndGgsIGZyYW1lU3RlcCk7XG4gIGNvbnN0IHdpbmRvd2VkU2lnbmFsID0gbXVsKGZyYW1lZFNpZ25hbCwgd2luZG93Rm4oZnJhbWVMZW5ndGgpKTtcbiAgcmV0dXJuIHJmZnQod2luZG93ZWRTaWduYWwsIGZmdExlbmd0aCk7XG59XG5leHBvcnQgY29uc3Qgc3RmdCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c3RmdF99KTtcbiJdfQ==