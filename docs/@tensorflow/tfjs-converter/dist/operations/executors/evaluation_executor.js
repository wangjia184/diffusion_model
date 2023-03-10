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
// tslint:disable-next-line: no-imports-from-dist
import * as tfOps from '@tensorflow/tfjs-core/dist/ops/ops_for_converter';
import { getParamValue } from './utils';
export const executeOp = (node, tensorMap, context, ops = tfOps) => {
    switch (node.op) {
        case 'LowerBound': {
            const sortedSequence = getParamValue('sortedSequence', node, tensorMap, context);
            const values = getParamValue('values', node, tensorMap, context);
            return [ops.lowerBound(sortedSequence, values)];
        }
        case 'TopKV2': {
            const x = getParamValue('x', node, tensorMap, context);
            const k = getParamValue('k', node, tensorMap, context);
            const sorted = getParamValue('sorted', node, tensorMap, context);
            const result = ops.topk(x, k, sorted);
            return [result.values, result.indices];
        }
        case 'UpperBound': {
            const sortedSequence = getParamValue('sortedSequence', node, tensorMap, context);
            const values = getParamValue('values', node, tensorMap, context);
            return [ops.upperBound(sortedSequence, values)];
        }
        case 'Unique': {
            const x = getParamValue('x', node, tensorMap, context);
            const result = ops.unique(x);
            return [result.values, result.indices];
        }
        case 'UniqueV2': {
            const x = getParamValue('x', node, tensorMap, context);
            const axis = getParamValue('axis', node, tensorMap, context);
            const result = ops.unique(x, axis);
            return [result.values, result.indices];
        }
        default:
            throw TypeError(`Node type ${node.op} is not implemented`);
    }
};
export const CATEGORY = 'evaluation';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXZhbHVhdGlvbl9leGVjdXRvci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29udmVydGVyL3NyYy9vcGVyYXRpb25zL2V4ZWN1dG9ycy9ldmFsdWF0aW9uX2V4ZWN1dG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUdILGlEQUFpRDtBQUNqRCxPQUFPLEtBQUssS0FBSyxNQUFNLGtEQUFrRCxDQUFDO0FBTTFFLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdEMsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUNsQixDQUFDLElBQVUsRUFBRSxTQUEwQixFQUFFLE9BQXlCLEVBQ2pFLEdBQUcsR0FBRyxLQUFLLEVBQ0MsRUFBRTtJQUNULFFBQVEsSUFBSSxDQUFDLEVBQUUsRUFBRTtRQUNmLEtBQUssWUFBWSxDQUFDLENBQUM7WUFDakIsTUFBTSxjQUFjLEdBQ2hCLGFBQWEsQ0FBQyxnQkFBZ0IsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FDbEQsQ0FBQztZQUNYLE1BQU0sTUFBTSxHQUNSLGFBQWEsQ0FBQyxRQUFRLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUNoRSxPQUFPLENBQUMsR0FBRyxDQUFDLFVBQVUsQ0FBQyxjQUFjLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQztTQUNqRDtRQUNELEtBQUssUUFBUSxDQUFDLENBQUM7WUFDYixNQUFNLENBQUMsR0FBRyxhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDakUsTUFBTSxDQUFDLEdBQUcsYUFBYSxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQ2pFLE1BQU0sTUFBTSxHQUNSLGFBQWEsQ0FBQyxRQUFRLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVksQ0FBQztZQUNqRSxNQUFNLE1BQU0sR0FBRyxHQUFHLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7WUFDdEMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1NBQ3hDO1FBQ0QsS0FBSyxZQUFZLENBQUMsQ0FBQztZQUNqQixNQUFNLGNBQWMsR0FDaEIsYUFBYSxDQUFDLGdCQUFnQixFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUNsRCxDQUFDO1lBQ1gsTUFBTSxNQUFNLEdBQ1IsYUFBYSxDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQ2hFLE9BQU8sQ0FBQyxHQUFHLENBQUMsVUFBVSxDQUFDLGNBQWMsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO1NBQ2pEO1FBQ0QsS0FBSyxRQUFRLENBQUMsQ0FBQztZQUNiLE1BQU0sQ0FBQyxHQUFHLGFBQWEsQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUNqRSxNQUFNLE1BQU0sR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzdCLE9BQU8sQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQztTQUN4QztRQUNELEtBQUssVUFBVSxDQUFDLENBQUM7WUFDZixNQUFNLENBQUMsR0FBRyxhQUFhLENBQUMsR0FBRyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDakUsTUFBTSxJQUFJLEdBQ04sYUFBYSxDQUFDLE1BQU0sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQzlELE1BQU0sTUFBTSxHQUFHLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO1lBQ25DLE9BQU8sQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQztTQUN4QztRQUNEO1lBQ0UsTUFBTSxTQUFTLENBQUMsYUFBYSxJQUFJLENBQUMsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO0tBQzlEO0FBQ0gsQ0FBQyxDQUFDO0FBRVYsTUFBTSxDQUFDLE1BQU0sUUFBUSxHQUFHLFlBQVksQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWltcG9ydHMtZnJvbS1kaXN0XG5pbXBvcnQgKiBhcyB0Zk9wcyBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUvZGlzdC9vcHMvb3BzX2Zvcl9jb252ZXJ0ZXInO1xuXG5pbXBvcnQge05hbWVkVGVuc29yc01hcH0gZnJvbSAnLi4vLi4vZGF0YS90eXBlcyc7XG5pbXBvcnQge0V4ZWN1dGlvbkNvbnRleHR9IGZyb20gJy4uLy4uL2V4ZWN1dG9yL2V4ZWN1dGlvbl9jb250ZXh0JztcbmltcG9ydCB7SW50ZXJuYWxPcEV4ZWN1dG9yLCBOb2RlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7Z2V0UGFyYW1WYWx1ZX0gZnJvbSAnLi91dGlscyc7XG5cbmV4cG9ydCBjb25zdCBleGVjdXRlT3A6IEludGVybmFsT3BFeGVjdXRvciA9XG4gICAgKG5vZGU6IE5vZGUsIHRlbnNvck1hcDogTmFtZWRUZW5zb3JzTWFwLCBjb250ZXh0OiBFeGVjdXRpb25Db250ZXh0LFxuICAgICBvcHMgPSB0Zk9wcyk6XG4gICAgICAgIFRlbnNvcltdID0+IHtcbiAgICAgICAgICBzd2l0Y2ggKG5vZGUub3ApIHtcbiAgICAgICAgICAgIGNhc2UgJ0xvd2VyQm91bmQnOiB7XG4gICAgICAgICAgICAgIGNvbnN0IHNvcnRlZFNlcXVlbmNlID1cbiAgICAgICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3NvcnRlZFNlcXVlbmNlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhc1xuICAgICAgICAgICAgICAgICAgVGVuc29yO1xuICAgICAgICAgICAgICBjb25zdCB2YWx1ZXMgPVxuICAgICAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndmFsdWVzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICAgICAgICAgIHJldHVybiBbb3BzLmxvd2VyQm91bmQoc29ydGVkU2VxdWVuY2UsIHZhbHVlcyldO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgY2FzZSAnVG9wS1YyJzoge1xuICAgICAgICAgICAgICBjb25zdCB4ID0gZ2V0UGFyYW1WYWx1ZSgneCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgICAgICAgICBjb25zdCBrID0gZ2V0UGFyYW1WYWx1ZSgnaycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgICAgICAgICBjb25zdCBzb3J0ZWQgPVxuICAgICAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnc29ydGVkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBib29sZWFuO1xuICAgICAgICAgICAgICBjb25zdCByZXN1bHQgPSBvcHMudG9wayh4LCBrLCBzb3J0ZWQpO1xuICAgICAgICAgICAgICByZXR1cm4gW3Jlc3VsdC52YWx1ZXMsIHJlc3VsdC5pbmRpY2VzXTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGNhc2UgJ1VwcGVyQm91bmQnOiB7XG4gICAgICAgICAgICAgIGNvbnN0IHNvcnRlZFNlcXVlbmNlID1cbiAgICAgICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3NvcnRlZFNlcXVlbmNlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhc1xuICAgICAgICAgICAgICAgICAgVGVuc29yO1xuICAgICAgICAgICAgICBjb25zdCB2YWx1ZXMgPVxuICAgICAgICAgICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndmFsdWVzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICAgICAgICAgIHJldHVybiBbb3BzLnVwcGVyQm91bmQoc29ydGVkU2VxdWVuY2UsIHZhbHVlcyldO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgY2FzZSAnVW5pcXVlJzoge1xuICAgICAgICAgICAgICBjb25zdCB4ID0gZ2V0UGFyYW1WYWx1ZSgneCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgICAgICAgICBjb25zdCByZXN1bHQgPSBvcHMudW5pcXVlKHgpO1xuICAgICAgICAgICAgICByZXR1cm4gW3Jlc3VsdC52YWx1ZXMsIHJlc3VsdC5pbmRpY2VzXTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGNhc2UgJ1VuaXF1ZVYyJzoge1xuICAgICAgICAgICAgICBjb25zdCB4ID0gZ2V0UGFyYW1WYWx1ZSgneCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgICAgICAgICBjb25zdCBheGlzID1cbiAgICAgICAgICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2F4aXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcjtcbiAgICAgICAgICAgICAgY29uc3QgcmVzdWx0ID0gb3BzLnVuaXF1ZSh4LCBheGlzKTtcbiAgICAgICAgICAgICAgcmV0dXJuIFtyZXN1bHQudmFsdWVzLCByZXN1bHQuaW5kaWNlc107XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBkZWZhdWx0OlxuICAgICAgICAgICAgICB0aHJvdyBUeXBlRXJyb3IoYE5vZGUgdHlwZSAke25vZGUub3B9IGlzIG5vdCBpbXBsZW1lbnRlZGApO1xuICAgICAgICAgIH1cbiAgICAgICAgfTtcblxuZXhwb3J0IGNvbnN0IENBVEVHT1JZID0gJ2V2YWx1YXRpb24nO1xuIl19