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
import { ENGINE } from '../engine';
import { assert, assertNonNegativeIntegerDimensions, flatten, inferDtype, isTypedArray, sizeFromShape, toTypedArray } from '../util';
/** This is shared code across all tensor creation methods. */
export function makeTensor(values, shape, inferredShape, dtype) {
    if (dtype == null) {
        dtype = inferDtype(values);
    }
    else if (dtype === 'complex64') {
        throw new Error(`Cannot construct a complex64 tensor directly. ` +
            `Please use tf.complex(real, imag).`);
    }
    if (typeof values === 'object' &&
        ('texture' in values ||
            ('buffer' in values && !(values.buffer instanceof ArrayBuffer)))) {
        if (dtype !== 'float32' && dtype !== 'int32') {
            throw new Error(`Creating tensor from GPU data only supports ` +
                `'float32'|'int32' dtype, while the dtype is ${dtype}.`);
        }
        return ENGINE.backend.createTensorFromGPUData(values, shape || inferredShape, dtype);
    }
    if (!isTypedArray(values) && !Array.isArray(values) &&
        typeof values !== 'number' && typeof values !== 'boolean' &&
        typeof values !== 'string') {
        throw new Error('values passed to tensor(values) must be a number/boolean/string or ' +
            'an array of numbers/booleans/strings, or a TypedArray');
    }
    // Verify that the shape matches the inferred shape.
    if (shape != null) {
        assertNonNegativeIntegerDimensions(shape);
        const providedSize = sizeFromShape(shape);
        const inferredSize = sizeFromShape(inferredShape);
        assert(providedSize === inferredSize, () => `Based on the provided shape, [${shape}], the tensor should have ` +
            `${providedSize} values but has ${inferredSize}`);
        for (let i = 0; i < inferredShape.length; ++i) {
            const inferred = inferredShape[i];
            const flatDimsDontMatch = i === inferredShape.length - 1 ?
                inferred !== sizeFromShape(shape.slice(i)) :
                true;
            assert(inferredShape[i] === shape[i] || !flatDimsDontMatch, () => `Error creating a new Tensor. Inferred shape ` +
                `(${inferredShape}) does not match the provided ` +
                `shape (${shape}). `);
        }
    }
    if (!isTypedArray(values) && !Array.isArray(values)) {
        values = [values];
    }
    shape = shape || inferredShape;
    values = dtype !== 'string' ?
        toTypedArray(values, dtype) :
        flatten(values, [], true);
    return ENGINE.makeTensor(values, shape, dtype);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVuc29yX29wc191dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvdGVuc29yX29wc191dGlsLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFJakMsT0FBTyxFQUFDLE1BQU0sRUFBRSxrQ0FBa0MsRUFBRSxPQUFPLEVBQUUsVUFBVSxFQUFFLFlBQVksRUFBRSxhQUFhLEVBQUUsWUFBWSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRW5JLDhEQUE4RDtBQUM5RCxNQUFNLFVBQVUsVUFBVSxDQUN0QixNQUF1QyxFQUFFLEtBQWUsRUFDeEQsYUFBdUIsRUFBRSxLQUFnQjtJQUMzQyxJQUFJLEtBQUssSUFBSSxJQUFJLEVBQUU7UUFDakIsS0FBSyxHQUFHLFVBQVUsQ0FBQyxNQUFNLENBQUMsQ0FBQztLQUM1QjtTQUFNLElBQUksS0FBSyxLQUFLLFdBQVcsRUFBRTtRQUNoQyxNQUFNLElBQUksS0FBSyxDQUNYLGdEQUFnRDtZQUNoRCxvQ0FBb0MsQ0FBQyxDQUFDO0tBQzNDO0lBRUQsSUFBSSxPQUFPLE1BQU0sS0FBSyxRQUFRO1FBQzFCLENBQUMsU0FBUyxJQUFJLE1BQU07WUFDbkIsQ0FBQyxRQUFRLElBQUksTUFBTSxJQUFJLENBQUMsQ0FBQyxNQUFNLENBQUMsTUFBTSxZQUFZLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRTtRQUNyRSxJQUFJLEtBQUssS0FBSyxTQUFTLElBQUksS0FBSyxLQUFLLE9BQU8sRUFBRTtZQUM1QyxNQUFNLElBQUksS0FBSyxDQUNYLDhDQUE4QztnQkFDOUMsK0NBQStDLEtBQUssR0FBRyxDQUFDLENBQUM7U0FDOUQ7UUFDRCxPQUFPLE1BQU0sQ0FBQyxPQUFPLENBQUMsdUJBQXVCLENBQ3pDLE1BQWdDLEVBQUUsS0FBSyxJQUFJLGFBQWEsRUFBRSxLQUFLLENBQUMsQ0FBQztLQUN0RTtJQUVELElBQUksQ0FBQyxZQUFZLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQztRQUMvQyxPQUFPLE1BQU0sS0FBSyxRQUFRLElBQUksT0FBTyxNQUFNLEtBQUssU0FBUztRQUN6RCxPQUFPLE1BQU0sS0FBSyxRQUFRLEVBQUU7UUFDOUIsTUFBTSxJQUFJLEtBQUssQ0FDWCxxRUFBcUU7WUFDckUsdURBQXVELENBQUMsQ0FBQztLQUM5RDtJQUNELG9EQUFvRDtJQUNwRCxJQUFJLEtBQUssSUFBSSxJQUFJLEVBQUU7UUFDakIsa0NBQWtDLENBQUMsS0FBSyxDQUFDLENBQUM7UUFFMUMsTUFBTSxZQUFZLEdBQUcsYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzFDLE1BQU0sWUFBWSxHQUFHLGFBQWEsQ0FBQyxhQUFhLENBQUMsQ0FBQztRQUNsRCxNQUFNLENBQ0YsWUFBWSxLQUFLLFlBQVksRUFDN0IsR0FBRyxFQUFFLENBQ0QsaUNBQWlDLEtBQUssNEJBQTRCO1lBQ2xFLEdBQUcsWUFBWSxtQkFBbUIsWUFBWSxFQUFFLENBQUMsQ0FBQztRQUUxRCxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsYUFBYSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtZQUM3QyxNQUFNLFFBQVEsR0FBRyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDbEMsTUFBTSxpQkFBaUIsR0FBRyxDQUFDLEtBQUssYUFBYSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztnQkFDdEQsUUFBUSxLQUFLLGFBQWEsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDNUMsSUFBSSxDQUFDO1lBQ1QsTUFBTSxDQUNGLGFBQWEsQ0FBQyxDQUFDLENBQUMsS0FBSyxLQUFLLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxpQkFBaUIsRUFDbkQsR0FBRyxFQUFFLENBQUMsOENBQThDO2dCQUNoRCxJQUFJLGFBQWEsZ0NBQWdDO2dCQUNqRCxVQUFVLEtBQUssS0FBSyxDQUFDLENBQUM7U0FDL0I7S0FDRjtJQUVELElBQUksQ0FBQyxZQUFZLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxFQUFFO1FBQ25ELE1BQU0sR0FBRyxDQUFDLE1BQU0sQ0FBYSxDQUFDO0tBQy9CO0lBRUQsS0FBSyxHQUFHLEtBQUssSUFBSSxhQUFhLENBQUM7SUFDL0IsTUFBTSxHQUFHLEtBQUssS0FBSyxRQUFRLENBQUMsQ0FBQztRQUN6QixZQUFZLENBQUMsTUFBTSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUM7UUFDN0IsT0FBTyxDQUFDLE1BQWtCLEVBQUUsRUFBRSxFQUFFLElBQUksQ0FBYSxDQUFDO0lBQ3RELE9BQU8sTUFBTSxDQUFDLFVBQVUsQ0FBQyxNQUFvQixFQUFFLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztBQUMvRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtUZW5zb3JMaWtlLCBUeXBlZEFycmF5LCBXZWJHTERhdGEsIFdlYkdQVURhdGF9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7RGF0YVR5cGV9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7YXNzZXJ0LCBhc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zLCBmbGF0dGVuLCBpbmZlckR0eXBlLCBpc1R5cGVkQXJyYXksIHNpemVGcm9tU2hhcGUsIHRvVHlwZWRBcnJheX0gZnJvbSAnLi4vdXRpbCc7XG5cbi8qKiBUaGlzIGlzIHNoYXJlZCBjb2RlIGFjcm9zcyBhbGwgdGVuc29yIGNyZWF0aW9uIG1ldGhvZHMuICovXG5leHBvcnQgZnVuY3Rpb24gbWFrZVRlbnNvcihcbiAgICB2YWx1ZXM6IFRlbnNvckxpa2V8V2ViR0xEYXRhfFdlYkdQVURhdGEsIHNoYXBlOiBudW1iZXJbXSxcbiAgICBpbmZlcnJlZFNoYXBlOiBudW1iZXJbXSwgZHR5cGU/OiBEYXRhVHlwZSk6IFRlbnNvciB7XG4gIGlmIChkdHlwZSA9PSBudWxsKSB7XG4gICAgZHR5cGUgPSBpbmZlckR0eXBlKHZhbHVlcyk7XG4gIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgQ2Fubm90IGNvbnN0cnVjdCBhIGNvbXBsZXg2NCB0ZW5zb3IgZGlyZWN0bHkuIGAgK1xuICAgICAgICBgUGxlYXNlIHVzZSB0Zi5jb21wbGV4KHJlYWwsIGltYWcpLmApO1xuICB9XG5cbiAgaWYgKHR5cGVvZiB2YWx1ZXMgPT09ICdvYmplY3QnICYmXG4gICAgICAoJ3RleHR1cmUnIGluIHZhbHVlcyB8fFxuICAgICAgICgnYnVmZmVyJyBpbiB2YWx1ZXMgJiYgISh2YWx1ZXMuYnVmZmVyIGluc3RhbmNlb2YgQXJyYXlCdWZmZXIpKSkpIHtcbiAgICBpZiAoZHR5cGUgIT09ICdmbG9hdDMyJyAmJiBkdHlwZSAhPT0gJ2ludDMyJykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBDcmVhdGluZyB0ZW5zb3IgZnJvbSBHUFUgZGF0YSBvbmx5IHN1cHBvcnRzIGAgK1xuICAgICAgICAgIGAnZmxvYXQzMid8J2ludDMyJyBkdHlwZSwgd2hpbGUgdGhlIGR0eXBlIGlzICR7ZHR5cGV9LmApO1xuICAgIH1cbiAgICByZXR1cm4gRU5HSU5FLmJhY2tlbmQuY3JlYXRlVGVuc29yRnJvbUdQVURhdGEoXG4gICAgICAgIHZhbHVlcyBhcyBXZWJHTERhdGEgfCBXZWJHUFVEYXRhLCBzaGFwZSB8fCBpbmZlcnJlZFNoYXBlLCBkdHlwZSk7XG4gIH1cblxuICBpZiAoIWlzVHlwZWRBcnJheSh2YWx1ZXMpICYmICFBcnJheS5pc0FycmF5KHZhbHVlcykgJiZcbiAgICAgIHR5cGVvZiB2YWx1ZXMgIT09ICdudW1iZXInICYmIHR5cGVvZiB2YWx1ZXMgIT09ICdib29sZWFuJyAmJlxuICAgICAgdHlwZW9mIHZhbHVlcyAhPT0gJ3N0cmluZycpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICd2YWx1ZXMgcGFzc2VkIHRvIHRlbnNvcih2YWx1ZXMpIG11c3QgYmUgYSBudW1iZXIvYm9vbGVhbi9zdHJpbmcgb3IgJyArXG4gICAgICAgICdhbiBhcnJheSBvZiBudW1iZXJzL2Jvb2xlYW5zL3N0cmluZ3MsIG9yIGEgVHlwZWRBcnJheScpO1xuICB9XG4gIC8vIFZlcmlmeSB0aGF0IHRoZSBzaGFwZSBtYXRjaGVzIHRoZSBpbmZlcnJlZCBzaGFwZS5cbiAgaWYgKHNoYXBlICE9IG51bGwpIHtcbiAgICBhc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zKHNoYXBlKTtcblxuICAgIGNvbnN0IHByb3ZpZGVkU2l6ZSA9IHNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIGNvbnN0IGluZmVycmVkU2l6ZSA9IHNpemVGcm9tU2hhcGUoaW5mZXJyZWRTaGFwZSk7XG4gICAgYXNzZXJ0KFxuICAgICAgICBwcm92aWRlZFNpemUgPT09IGluZmVycmVkU2l6ZSxcbiAgICAgICAgKCkgPT5cbiAgICAgICAgICAgIGBCYXNlZCBvbiB0aGUgcHJvdmlkZWQgc2hhcGUsIFske3NoYXBlfV0sIHRoZSB0ZW5zb3Igc2hvdWxkIGhhdmUgYCArXG4gICAgICAgICAgICBgJHtwcm92aWRlZFNpemV9IHZhbHVlcyBidXQgaGFzICR7aW5mZXJyZWRTaXplfWApO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBpbmZlcnJlZFNoYXBlLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCBpbmZlcnJlZCA9IGluZmVycmVkU2hhcGVbaV07XG4gICAgICBjb25zdCBmbGF0RGltc0RvbnRNYXRjaCA9IGkgPT09IGluZmVycmVkU2hhcGUubGVuZ3RoIC0gMSA/XG4gICAgICAgICAgaW5mZXJyZWQgIT09IHNpemVGcm9tU2hhcGUoc2hhcGUuc2xpY2UoaSkpIDpcbiAgICAgICAgICB0cnVlO1xuICAgICAgYXNzZXJ0KFxuICAgICAgICAgIGluZmVycmVkU2hhcGVbaV0gPT09IHNoYXBlW2ldIHx8ICFmbGF0RGltc0RvbnRNYXRjaCxcbiAgICAgICAgICAoKSA9PiBgRXJyb3IgY3JlYXRpbmcgYSBuZXcgVGVuc29yLiBJbmZlcnJlZCBzaGFwZSBgICtcbiAgICAgICAgICAgICAgYCgke2luZmVycmVkU2hhcGV9KSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgYCArXG4gICAgICAgICAgICAgIGBzaGFwZSAoJHtzaGFwZX0pLiBgKTtcbiAgICB9XG4gIH1cblxuICBpZiAoIWlzVHlwZWRBcnJheSh2YWx1ZXMpICYmICFBcnJheS5pc0FycmF5KHZhbHVlcykpIHtcbiAgICB2YWx1ZXMgPSBbdmFsdWVzXSBhcyBudW1iZXJbXTtcbiAgfVxuXG4gIHNoYXBlID0gc2hhcGUgfHwgaW5mZXJyZWRTaGFwZTtcbiAgdmFsdWVzID0gZHR5cGUgIT09ICdzdHJpbmcnID9cbiAgICAgIHRvVHlwZWRBcnJheSh2YWx1ZXMsIGR0eXBlKSA6XG4gICAgICBmbGF0dGVuKHZhbHVlcyBhcyBzdHJpbmdbXSwgW10sIHRydWUpIGFzIHN0cmluZ1tdO1xuICByZXR1cm4gRU5HSU5FLm1ha2VUZW5zb3IodmFsdWVzIGFzIFR5cGVkQXJyYXksIHNoYXBlLCBkdHlwZSk7XG59XG4iXX0=