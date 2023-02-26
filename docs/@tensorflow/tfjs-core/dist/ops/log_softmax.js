/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import { customGrad } from '../gradients';
import { convertToTensor } from '../tensor_util_env';
import { cast } from './cast';
import { exp } from './exp';
import { log } from './log';
import { max } from './max';
import { mul } from './mul';
import { op } from './operation';
import { sub } from './sub';
import { sum } from './sum';
/**
 * Computes the log softmax.
 *
 * ```js
 * const a = tf.tensor1d([1, 2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * ```js
 * const a = tf.tensor2d([2, 4, 6, 1, 2, 3], [2, 3]);
 *
 * a.logSoftmax().print();  // or tf.logSoftmax(a)
 * ```
 *
 * @param logits The logits array.
 * @param axis The dimension softmax would be performed on. Defaults to `-1`
 *     which indicates the last dimension.
 *
 * @doc {heading: 'Operations', subheading: 'Normalization'}
 */
function logSoftmax_(logits, axis = -1) {
    const $logits = convertToTensor(logits, 'logits', 'logSoftmax');
    if (axis === -1) {
        axis = $logits.rank - 1;
    }
    if (axis !== $logits.rank - 1) {
        throw Error('Log Softmax along a non-last dimension is not yet supported. ' +
            `Logits was rank ${$logits.rank} and axis was ${axis}`);
    }
    // const forward: ForwardFunc<Tensor> = (backend, save) => {
    //   const keepDims = true;
    //   const xMax = max(logits, axis, true);
    //   const shifted = sub(logits, xMax);
    //   const value =
    //       sub(cast(shifted, 'float32'), log(sum(exp(shifted), axis,
    //       keepDims)));
    //   save([value]);
    //   return value;
    // };
    // Use a custom gradient for numerical stability.
    const customOp = customGrad((logits, save) => {
        const keepDims = true;
        const xMax = max(logits, axis, true);
        const shifted = sub(logits, xMax);
        const value = sub(cast(shifted, 'float32'), log(sum(exp(shifted), axis, keepDims)));
        save([value]);
        const gradFunc = (dy, saved) => {
            const [value] = saved;
            const keepDims = true;
            const softmax = exp(value);
            return sub(dy, mul(sum(dy, axis, keepDims), softmax));
        };
        return { value, gradFunc };
    });
    return customOp($logits);
    // TODO Use Engine.runKernel when CPU/WebGL/WASM backends implement this.
    // const inputs: LogSoftmaxInputs = {logits: $logits};
    // const attrs: LogSoftmaxAttrs = {axis};
    // return ENGINE.runKernel(
    //            LogSoftmax, inputs as unknown as NamedTensorMap,
    //            attrs as unknown as NamedAttrMap);
}
export const logSoftmax = /* @__PURE__ */ op({ logSoftmax_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9nX3NvZnRtYXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9sb2dfc29mdG1heC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBSXhDLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzVCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFFMUI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBb0JHO0FBQ0gsU0FBUyxXQUFXLENBQW1CLE1BQW9CLEVBQUUsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNwRSxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxZQUFZLENBQUMsQ0FBQztJQUVoRSxJQUFJLElBQUksS0FBSyxDQUFDLENBQUMsRUFBRTtRQUNmLElBQUksR0FBRyxPQUFPLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztLQUN6QjtJQUNELElBQUksSUFBSSxLQUFLLE9BQU8sQ0FBQyxJQUFJLEdBQUcsQ0FBQyxFQUFFO1FBQzdCLE1BQU0sS0FBSyxDQUNQLCtEQUErRDtZQUMvRCxtQkFBbUIsT0FBTyxDQUFDLElBQUksaUJBQWlCLElBQUksRUFBRSxDQUFDLENBQUM7S0FDN0Q7SUFFRCw0REFBNEQ7SUFDNUQsMkJBQTJCO0lBQzNCLDBDQUEwQztJQUMxQyx1Q0FBdUM7SUFDdkMsa0JBQWtCO0lBQ2xCLGtFQUFrRTtJQUNsRSxxQkFBcUI7SUFDckIsbUJBQW1CO0lBQ25CLGtCQUFrQjtJQUNsQixLQUFLO0lBRUwsaURBQWlEO0lBQ2pELE1BQU0sUUFBUSxHQUFHLFVBQVUsQ0FBQyxDQUFDLE1BQWMsRUFBRSxJQUFrQixFQUFFLEVBQUU7UUFDakUsTUFBTSxRQUFRLEdBQUcsSUFBSSxDQUFDO1FBQ3RCLE1BQU0sSUFBSSxHQUFHLEdBQUcsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO1FBQ3JDLE1BQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQyxNQUFNLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDbEMsTUFBTSxLQUFLLEdBQ1AsR0FBRyxDQUFDLElBQUksQ0FBQyxPQUFPLEVBQUUsU0FBUyxDQUFDLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsT0FBTyxDQUFDLEVBQUUsSUFBSSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMxRSxJQUFJLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1FBRWQsTUFBTSxRQUFRLEdBQUcsQ0FBQyxFQUFVLEVBQUUsS0FBZSxFQUFFLEVBQUU7WUFDL0MsTUFBTSxDQUFDLEtBQUssQ0FBQyxHQUFHLEtBQUssQ0FBQztZQUN0QixNQUFNLFFBQVEsR0FBRyxJQUFJLENBQUM7WUFDdEIsTUFBTSxPQUFPLEdBQUcsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQzNCLE9BQU8sR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsUUFBUSxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUN4RCxDQUFDLENBQUM7UUFDRixPQUFPLEVBQUMsS0FBSyxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQzNCLENBQUMsQ0FBQyxDQUFDO0lBRUgsT0FBTyxRQUFRLENBQUMsT0FBTyxDQUFNLENBQUM7SUFFOUIseUVBQXlFO0lBQ3pFLHNEQUFzRDtJQUN0RCx5Q0FBeUM7SUFDekMsMkJBQTJCO0lBQzNCLDhEQUE4RDtJQUM5RCxnREFBZ0Q7QUFDbEQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFVBQVUsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsV0FBVyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtjdXN0b21HcmFkfSBmcm9tICcuLi9ncmFkaWVudHMnO1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7R3JhZFNhdmVGdW5jfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtjYXN0fSBmcm9tICcuL2Nhc3QnO1xuaW1wb3J0IHtleHB9IGZyb20gJy4vZXhwJztcbmltcG9ydCB7bG9nfSBmcm9tICcuL2xvZyc7XG5pbXBvcnQge21heH0gZnJvbSAnLi9tYXgnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4vbXVsJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7c3VifSBmcm9tICcuL3N1Yic7XG5pbXBvcnQge3N1bX0gZnJvbSAnLi9zdW0nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBsb2cgc29mdG1heC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKlxuICogYS5sb2dTb2Z0bWF4KCkucHJpbnQoKTsgIC8vIG9yIHRmLmxvZ1NvZnRtYXgoYSlcbiAqIGBgYFxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYudGVuc29yMmQoWzIsIDQsIDYsIDEsIDIsIDNdLCBbMiwgM10pO1xuICpcbiAqIGEubG9nU29mdG1heCgpLnByaW50KCk7ICAvLyBvciB0Zi5sb2dTb2Z0bWF4KGEpXG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gbG9naXRzIFRoZSBsb2dpdHMgYXJyYXkuXG4gKiBAcGFyYW0gYXhpcyBUaGUgZGltZW5zaW9uIHNvZnRtYXggd291bGQgYmUgcGVyZm9ybWVkIG9uLiBEZWZhdWx0cyB0byBgLTFgXG4gKiAgICAgd2hpY2ggaW5kaWNhdGVzIHRoZSBsYXN0IGRpbWVuc2lvbi5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdOb3JtYWxpemF0aW9uJ31cbiAqL1xuZnVuY3Rpb24gbG9nU29mdG1heF88VCBleHRlbmRzIFRlbnNvcj4obG9naXRzOiBUfFRlbnNvckxpa2UsIGF4aXMgPSAtMSk6IFQge1xuICBjb25zdCAkbG9naXRzID0gY29udmVydFRvVGVuc29yKGxvZ2l0cywgJ2xvZ2l0cycsICdsb2dTb2Z0bWF4Jyk7XG5cbiAgaWYgKGF4aXMgPT09IC0xKSB7XG4gICAgYXhpcyA9ICRsb2dpdHMucmFuayAtIDE7XG4gIH1cbiAgaWYgKGF4aXMgIT09ICRsb2dpdHMucmFuayAtIDEpIHtcbiAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgJ0xvZyBTb2Z0bWF4IGFsb25nIGEgbm9uLWxhc3QgZGltZW5zaW9uIGlzIG5vdCB5ZXQgc3VwcG9ydGVkLiAnICtcbiAgICAgICAgYExvZ2l0cyB3YXMgcmFuayAkeyRsb2dpdHMucmFua30gYW5kIGF4aXMgd2FzICR7YXhpc31gKTtcbiAgfVxuXG4gIC8vIGNvbnN0IGZvcndhcmQ6IEZvcndhcmRGdW5jPFRlbnNvcj4gPSAoYmFja2VuZCwgc2F2ZSkgPT4ge1xuICAvLyAgIGNvbnN0IGtlZXBEaW1zID0gdHJ1ZTtcbiAgLy8gICBjb25zdCB4TWF4ID0gbWF4KGxvZ2l0cywgYXhpcywgdHJ1ZSk7XG4gIC8vICAgY29uc3Qgc2hpZnRlZCA9IHN1Yihsb2dpdHMsIHhNYXgpO1xuICAvLyAgIGNvbnN0IHZhbHVlID1cbiAgLy8gICAgICAgc3ViKGNhc3Qoc2hpZnRlZCwgJ2Zsb2F0MzInKSwgbG9nKHN1bShleHAoc2hpZnRlZCksIGF4aXMsXG4gIC8vICAgICAgIGtlZXBEaW1zKSkpO1xuICAvLyAgIHNhdmUoW3ZhbHVlXSk7XG4gIC8vICAgcmV0dXJuIHZhbHVlO1xuICAvLyB9O1xuXG4gIC8vIFVzZSBhIGN1c3RvbSBncmFkaWVudCBmb3IgbnVtZXJpY2FsIHN0YWJpbGl0eS5cbiAgY29uc3QgY3VzdG9tT3AgPSBjdXN0b21HcmFkKChsb2dpdHM6IFRlbnNvciwgc2F2ZTogR3JhZFNhdmVGdW5jKSA9PiB7XG4gICAgY29uc3Qga2VlcERpbXMgPSB0cnVlO1xuICAgIGNvbnN0IHhNYXggPSBtYXgobG9naXRzLCBheGlzLCB0cnVlKTtcbiAgICBjb25zdCBzaGlmdGVkID0gc3ViKGxvZ2l0cywgeE1heCk7XG4gICAgY29uc3QgdmFsdWUgPVxuICAgICAgICBzdWIoY2FzdChzaGlmdGVkLCAnZmxvYXQzMicpLCBsb2coc3VtKGV4cChzaGlmdGVkKSwgYXhpcywga2VlcERpbXMpKSk7XG4gICAgc2F2ZShbdmFsdWVdKTtcblxuICAgIGNvbnN0IGdyYWRGdW5jID0gKGR5OiBUZW5zb3IsIHNhdmVkOiBUZW5zb3JbXSkgPT4ge1xuICAgICAgY29uc3QgW3ZhbHVlXSA9IHNhdmVkO1xuICAgICAgY29uc3Qga2VlcERpbXMgPSB0cnVlO1xuICAgICAgY29uc3Qgc29mdG1heCA9IGV4cCh2YWx1ZSk7XG4gICAgICByZXR1cm4gc3ViKGR5LCBtdWwoc3VtKGR5LCBheGlzLCBrZWVwRGltcyksIHNvZnRtYXgpKTtcbiAgICB9O1xuICAgIHJldHVybiB7dmFsdWUsIGdyYWRGdW5jfTtcbiAgfSk7XG5cbiAgcmV0dXJuIGN1c3RvbU9wKCRsb2dpdHMpIGFzIFQ7XG5cbiAgLy8gVE9ETyBVc2UgRW5naW5lLnJ1bktlcm5lbCB3aGVuIENQVS9XZWJHTC9XQVNNIGJhY2tlbmRzIGltcGxlbWVudCB0aGlzLlxuICAvLyBjb25zdCBpbnB1dHM6IExvZ1NvZnRtYXhJbnB1dHMgPSB7bG9naXRzOiAkbG9naXRzfTtcbiAgLy8gY29uc3QgYXR0cnM6IExvZ1NvZnRtYXhBdHRycyA9IHtheGlzfTtcbiAgLy8gcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gIC8vICAgICAgICAgICAgTG9nU29mdG1heCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gIC8vICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgbG9nU29mdG1heCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bG9nU29mdG1heF99KTtcbiJdfQ==