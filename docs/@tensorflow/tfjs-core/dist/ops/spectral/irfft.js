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
 * =============================================================================
 */
import { complex } from '../complex';
import { concat } from '../concat';
import { imag } from '../imag';
import { mul } from '../mul';
import { op } from '../operation';
import { real } from '../real';
import { reshape } from '../reshape';
import { reverse } from '../reverse';
import { scalar } from '../scalar';
import { slice } from '../slice';
import { ifft } from './ifft';
/**
 * Inversed real value input fast Fourier transform.
 *
 * Computes the 1-dimensional inversed discrete Fourier transform over the
 * inner-most dimension of the real input.
 *
 * ```js
 * const real = tf.tensor1d([1, 2, 3]);
 * const imag = tf.tensor1d([0, 0, 0]);
 * const x = tf.complex(real, imag);
 *
 * x.irfft().print();
 * ```
 * @param input The real value input to compute an irfft over.
 *
 * @doc {heading: 'Operations', subheading: 'Spectral', namespace: 'spectral'}
 */
function irfft_(input) {
    const innerDimensionSize = input.shape[input.shape.length - 1];
    const batch = input.size / innerDimensionSize;
    let ret;
    if (innerDimensionSize <= 2) {
        const complexInput = reshape(input, [batch, innerDimensionSize]);
        ret = ifft(complexInput);
    }
    else {
        // The length of unique components of the DFT of a real-valued signal
        // is 2 * (input_len - 1)
        const outputShape = [batch, 2 * (innerDimensionSize - 1)];
        const realInput = reshape(real(input), [batch, innerDimensionSize]);
        const imagInput = reshape(imag(input), [batch, innerDimensionSize]);
        const realConjugate = reverse(slice(realInput, [0, 1], [batch, innerDimensionSize - 2]), 1);
        const imagConjugate = mul(reverse(slice(imagInput, [0, 1], [batch, innerDimensionSize - 2]), 1), scalar(-1));
        const r = concat([realInput, realConjugate], 1);
        const i = concat([imagInput, imagConjugate], 1);
        const complexInput = reshape(complex(r, i), [outputShape[0], outputShape[1]]);
        ret = ifft(complexInput);
    }
    ret = real(ret);
    // reshape the result if the input is 3D tensor.
    if (input.rank === 3 && input.shape[0] !== 0) {
        const temp = ret;
        const batch = input.shape[0];
        ret = reshape(ret, [batch, ret.shape[0] / batch, ret.shape[1]]);
        temp.dispose();
    }
    return ret;
}
export const irfft = /* @__PURE__ */ op({ irfft_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaXJmZnQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zcGVjdHJhbC9pcmZmdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ25DLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUM3QixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDaEMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUM3QixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBQ25DLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBRS9CLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFNUI7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFDSCxTQUFTLE1BQU0sQ0FBQyxLQUFhO0lBQzNCLE1BQU0sa0JBQWtCLEdBQUcsS0FBSyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztJQUMvRCxNQUFNLEtBQUssR0FBRyxLQUFLLENBQUMsSUFBSSxHQUFHLGtCQUFrQixDQUFDO0lBQzlDLElBQUksR0FBVyxDQUFDO0lBQ2hCLElBQUksa0JBQWtCLElBQUksQ0FBQyxFQUFFO1FBQzNCLE1BQU0sWUFBWSxHQUFHLE9BQU8sQ0FBQyxLQUFLLEVBQUUsQ0FBQyxLQUFLLEVBQUUsa0JBQWtCLENBQUMsQ0FBQyxDQUFDO1FBQ2pFLEdBQUcsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUM7S0FDMUI7U0FBTTtRQUNMLHFFQUFxRTtRQUNyRSx5QkFBeUI7UUFDekIsTUFBTSxXQUFXLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxHQUFHLENBQUMsa0JBQWtCLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMxRCxNQUFNLFNBQVMsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLGtCQUFrQixDQUFDLENBQUMsQ0FBQztRQUNwRSxNQUFNLFNBQVMsR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLGtCQUFrQixDQUFDLENBQUMsQ0FBQztRQUVwRSxNQUFNLGFBQWEsR0FDZixPQUFPLENBQUMsS0FBSyxDQUFDLFNBQVMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxrQkFBa0IsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzFFLE1BQU0sYUFBYSxHQUFhLEdBQUcsQ0FDL0IsT0FBTyxDQUFDLEtBQUssQ0FBQyxTQUFTLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxLQUFLLEVBQUUsa0JBQWtCLEdBQUcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFDckUsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVoQixNQUFNLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQyxTQUFTLEVBQUUsYUFBYSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDaEQsTUFBTSxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsU0FBUyxFQUFFLGFBQWEsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ2hELE1BQU0sWUFBWSxHQUNkLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDN0QsR0FBRyxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQztLQUMxQjtJQUNELEdBQUcsR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDaEIsZ0RBQWdEO0lBQ2hELElBQUksS0FBSyxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLEVBQUU7UUFDNUMsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDO1FBQ2pCLE1BQU0sS0FBSyxHQUFHLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDN0IsR0FBRyxHQUFHLE9BQU8sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxLQUFLLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxLQUFLLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDaEUsSUFBSSxDQUFDLE9BQU8sRUFBRSxDQUFDO0tBQ2hCO0lBQ0QsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sS0FBSyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxNQUFNLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvciwgVGVuc29yMkR9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbXBsZXh9IGZyb20gJy4uL2NvbXBsZXgnO1xuaW1wb3J0IHtjb25jYXR9IGZyb20gJy4uL2NvbmNhdCc7XG5pbXBvcnQge2ltYWd9IGZyb20gJy4uL2ltYWcnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4uL211bCc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZWFsfSBmcm9tICcuLi9yZWFsJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi4vcmVzaGFwZSc7XG5pbXBvcnQge3JldmVyc2V9IGZyb20gJy4uL3JldmVyc2UnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4uL3NjYWxhcic7XG5pbXBvcnQge3NsaWNlfSBmcm9tICcuLi9zbGljZSc7XG5cbmltcG9ydCB7aWZmdH0gZnJvbSAnLi9pZmZ0JztcblxuLyoqXG4gKiBJbnZlcnNlZCByZWFsIHZhbHVlIGlucHV0IGZhc3QgRm91cmllciB0cmFuc2Zvcm0uXG4gKlxuICogQ29tcHV0ZXMgdGhlIDEtZGltZW5zaW9uYWwgaW52ZXJzZWQgZGlzY3JldGUgRm91cmllciB0cmFuc2Zvcm0gb3ZlciB0aGVcbiAqIGlubmVyLW1vc3QgZGltZW5zaW9uIG9mIHRoZSByZWFsIGlucHV0LlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCByZWFsID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAqIGNvbnN0IGltYWcgPSB0Zi50ZW5zb3IxZChbMCwgMCwgMF0pO1xuICogY29uc3QgeCA9IHRmLmNvbXBsZXgocmVhbCwgaW1hZyk7XG4gKlxuICogeC5pcmZmdCgpLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSBpbnB1dCBUaGUgcmVhbCB2YWx1ZSBpbnB1dCB0byBjb21wdXRlIGFuIGlyZmZ0IG92ZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU3BlY3RyYWwnLCBuYW1lc3BhY2U6ICdzcGVjdHJhbCd9XG4gKi9cbmZ1bmN0aW9uIGlyZmZ0XyhpbnB1dDogVGVuc29yKTogVGVuc29yIHtcbiAgY29uc3QgaW5uZXJEaW1lbnNpb25TaXplID0gaW5wdXQuc2hhcGVbaW5wdXQuc2hhcGUubGVuZ3RoIC0gMV07XG4gIGNvbnN0IGJhdGNoID0gaW5wdXQuc2l6ZSAvIGlubmVyRGltZW5zaW9uU2l6ZTtcbiAgbGV0IHJldDogVGVuc29yO1xuICBpZiAoaW5uZXJEaW1lbnNpb25TaXplIDw9IDIpIHtcbiAgICBjb25zdCBjb21wbGV4SW5wdXQgPSByZXNoYXBlKGlucHV0LCBbYmF0Y2gsIGlubmVyRGltZW5zaW9uU2l6ZV0pO1xuICAgIHJldCA9IGlmZnQoY29tcGxleElucHV0KTtcbiAgfSBlbHNlIHtcbiAgICAvLyBUaGUgbGVuZ3RoIG9mIHVuaXF1ZSBjb21wb25lbnRzIG9mIHRoZSBERlQgb2YgYSByZWFsLXZhbHVlZCBzaWduYWxcbiAgICAvLyBpcyAyICogKGlucHV0X2xlbiAtIDEpXG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPSBbYmF0Y2gsIDIgKiAoaW5uZXJEaW1lbnNpb25TaXplIC0gMSldO1xuICAgIGNvbnN0IHJlYWxJbnB1dCA9IHJlc2hhcGUocmVhbChpbnB1dCksIFtiYXRjaCwgaW5uZXJEaW1lbnNpb25TaXplXSk7XG4gICAgY29uc3QgaW1hZ0lucHV0ID0gcmVzaGFwZShpbWFnKGlucHV0KSwgW2JhdGNoLCBpbm5lckRpbWVuc2lvblNpemVdKTtcblxuICAgIGNvbnN0IHJlYWxDb25qdWdhdGUgPVxuICAgICAgICByZXZlcnNlKHNsaWNlKHJlYWxJbnB1dCwgWzAsIDFdLCBbYmF0Y2gsIGlubmVyRGltZW5zaW9uU2l6ZSAtIDJdKSwgMSk7XG4gICAgY29uc3QgaW1hZ0Nvbmp1Z2F0ZTogVGVuc29yMkQgPSBtdWwoXG4gICAgICAgIHJldmVyc2Uoc2xpY2UoaW1hZ0lucHV0LCBbMCwgMV0sIFtiYXRjaCwgaW5uZXJEaW1lbnNpb25TaXplIC0gMl0pLCAxKSxcbiAgICAgICAgc2NhbGFyKC0xKSk7XG5cbiAgICBjb25zdCByID0gY29uY2F0KFtyZWFsSW5wdXQsIHJlYWxDb25qdWdhdGVdLCAxKTtcbiAgICBjb25zdCBpID0gY29uY2F0KFtpbWFnSW5wdXQsIGltYWdDb25qdWdhdGVdLCAxKTtcbiAgICBjb25zdCBjb21wbGV4SW5wdXQgPVxuICAgICAgICByZXNoYXBlKGNvbXBsZXgociwgaSksIFtvdXRwdXRTaGFwZVswXSwgb3V0cHV0U2hhcGVbMV1dKTtcbiAgICByZXQgPSBpZmZ0KGNvbXBsZXhJbnB1dCk7XG4gIH1cbiAgcmV0ID0gcmVhbChyZXQpO1xuICAvLyByZXNoYXBlIHRoZSByZXN1bHQgaWYgdGhlIGlucHV0IGlzIDNEIHRlbnNvci5cbiAgaWYgKGlucHV0LnJhbmsgPT09IDMgJiYgaW5wdXQuc2hhcGVbMF0gIT09IDApIHtcbiAgICBjb25zdCB0ZW1wID0gcmV0O1xuICAgIGNvbnN0IGJhdGNoID0gaW5wdXQuc2hhcGVbMF07XG4gICAgcmV0ID0gcmVzaGFwZShyZXQsIFtiYXRjaCwgcmV0LnNoYXBlWzBdIC8gYmF0Y2gsIHJldC5zaGFwZVsxXV0pO1xuICAgIHRlbXAuZGlzcG9zZSgpO1xuICB9XG4gIHJldHVybiByZXQ7XG59XG5cbmV4cG9ydCBjb25zdCBpcmZmdCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7aXJmZnRffSk7XG4iXX0=