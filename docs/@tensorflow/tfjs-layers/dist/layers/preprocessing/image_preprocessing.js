/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
import { Layer } from '../../engine/topology';
import { serialization, mul, add, tidy } from '@tensorflow/tfjs-core';
import { getExactlyOneTensor } from '../../utils/types_utils';
import * as K from '../../backend/tfjs_backend';
/**
 * Preprocessing Rescaling Layer
 *
 * This rescales images by a scaling and offset factor
 */
export class Rescaling extends Layer {
    constructor(args) {
        super(args);
        this.scale = args.scale;
        if (args.offset) {
            this.offset = args.offset;
        }
        else {
            this.offset = 0;
        }
    }
    getConfig() {
        const config = {
            'scale': this.scale,
            'offset': this.offset
        };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }
    call(inputs, kwargs) {
        return tidy(() => {
            inputs = getExactlyOneTensor(inputs);
            if (inputs.dtype !== 'float32') {
                inputs = K.cast(inputs, 'float32');
            }
            return add(mul(inputs, this.scale), this.offset);
        });
    }
}
/** @nocollapse */
Rescaling.className = 'Rescaling';
serialization.registerClass(Rescaling);
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW1hZ2VfcHJlcHJvY2Vzc2luZy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtbGF5ZXJzL3NyYy9sYXllcnMvcHJlcHJvY2Vzc2luZy9pbWFnZV9wcmVwcm9jZXNzaW5nLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBRUgsT0FBTyxFQUFZLEtBQUssRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ3ZELE9BQU8sRUFBRSxhQUFhLEVBQVUsR0FBRyxFQUFFLEdBQUcsRUFBRSxJQUFJLEVBQUUsTUFBTSx1QkFBdUIsQ0FBQztBQUM5RSxPQUFPLEVBQUUsbUJBQW1CLEVBQUUsTUFBTSx5QkFBeUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssQ0FBQyxNQUFNLDRCQUE0QixDQUFDO0FBUWhEOzs7O0dBSUc7QUFDSCxNQUFNLE9BQU8sU0FBVSxTQUFRLEtBQUs7SUFLbEMsWUFBWSxJQUFtQjtRQUM3QixLQUFLLENBQUMsSUFBSSxDQUFDLENBQUM7UUFFWixJQUFJLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUM7UUFFeEIsSUFBRyxJQUFJLENBQUMsTUFBTSxFQUFFO1lBQ2hCLElBQUksQ0FBQyxNQUFNLEdBQUcsSUFBSSxDQUFDLE1BQU0sQ0FBQztTQUN6QjthQUFNO1lBQ0wsSUFBSSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7U0FDakI7SUFDSCxDQUFDO0lBRVEsU0FBUztRQUNoQixNQUFNLE1BQU0sR0FBNkI7WUFDdkMsT0FBTyxFQUFFLElBQUksQ0FBQyxLQUFLO1lBQ25CLFFBQVEsRUFBRSxJQUFJLENBQUMsTUFBTTtTQUN0QixDQUFDO1FBRUYsTUFBTSxVQUFVLEdBQUcsS0FBSyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQ3JDLE1BQU0sQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLFVBQVUsQ0FBQyxDQUFDO1FBQ2xDLE9BQU8sTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFFUSxJQUFJLENBQUMsTUFBdUIsRUFBRSxNQUFjO1FBQ25ELE9BQU8sSUFBSSxDQUFDLEdBQUcsRUFBRTtZQUNmLE1BQU0sR0FBRyxtQkFBbUIsQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUNyQyxJQUFHLE1BQU0sQ0FBQyxLQUFLLEtBQUssU0FBUyxFQUFFO2dCQUMzQixNQUFNLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLEVBQUUsU0FBUyxDQUFDLENBQUM7YUFDdEM7WUFDRCxPQUFPLEdBQUcsQ0FBQyxHQUFHLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyxLQUFLLENBQUMsRUFBRSxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDbkQsQ0FBQyxDQUFDLENBQUM7SUFDTCxDQUFDOztBQW5DRCxrQkFBa0I7QUFDWCxtQkFBUyxHQUFHLFdBQVcsQ0FBQztBQXFDakMsYUFBYSxDQUFDLGFBQWEsQ0FBQyxTQUFTLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIENvZGVTbWl0aCBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7TGF5ZXJBcmdzLCBMYXllcn0gZnJvbSAnLi4vLi4vZW5naW5lL3RvcG9sb2d5JztcbmltcG9ydCB7IHNlcmlhbGl6YXRpb24sIFRlbnNvciwgbXVsLCBhZGQsIHRpZHkgfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHsgZ2V0RXhhY3RseU9uZVRlbnNvciB9IGZyb20gJy4uLy4uL3V0aWxzL3R5cGVzX3V0aWxzJztcbmltcG9ydCAqIGFzIEsgZnJvbSAnLi4vLi4vYmFja2VuZC90ZmpzX2JhY2tlbmQnO1xuaW1wb3J0IHsgS3dhcmdzIH0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuXG5leHBvcnQgZGVjbGFyZSBpbnRlcmZhY2UgUmVzY2FsaW5nQXJncyBleHRlbmRzIExheWVyQXJncyB7XG4gIHNjYWxlOiBudW1iZXI7XG4gIG9mZnNldD86IG51bWJlcjtcbn1cblxuLyoqXG4gKiBQcmVwcm9jZXNzaW5nIFJlc2NhbGluZyBMYXllclxuICpcbiAqIFRoaXMgcmVzY2FsZXMgaW1hZ2VzIGJ5IGEgc2NhbGluZyBhbmQgb2Zmc2V0IGZhY3RvclxuICovXG5leHBvcnQgY2xhc3MgUmVzY2FsaW5nIGV4dGVuZHMgTGF5ZXIge1xuICAvKiogQG5vY29sbGFwc2UgKi9cbiAgc3RhdGljIGNsYXNzTmFtZSA9ICdSZXNjYWxpbmcnO1xuICBwcml2YXRlIHJlYWRvbmx5IHNjYWxlOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgb2Zmc2V0OiBudW1iZXI7XG4gIGNvbnN0cnVjdG9yKGFyZ3M6IFJlc2NhbGluZ0FyZ3MpIHtcbiAgICBzdXBlcihhcmdzKTtcblxuICAgIHRoaXMuc2NhbGUgPSBhcmdzLnNjYWxlO1xuXG4gICAgaWYoYXJncy5vZmZzZXQpIHtcbiAgICB0aGlzLm9mZnNldCA9IGFyZ3Mub2Zmc2V0O1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLm9mZnNldCA9IDA7XG4gICAgfVxuICB9XG5cbiAgb3ZlcnJpZGUgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgY29uc3QgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7XG4gICAgICAnc2NhbGUnOiB0aGlzLnNjYWxlLFxuICAgICAgJ29mZnNldCc6IHRoaXMub2Zmc2V0XG4gICAgfTtcblxuICAgIGNvbnN0IGJhc2VDb25maWcgPSBzdXBlci5nZXRDb25maWcoKTtcbiAgICBPYmplY3QuYXNzaWduKGNvbmZpZywgYmFzZUNvbmZpZyk7XG4gICAgcmV0dXJuIGNvbmZpZztcbiAgfVxuXG4gIG92ZXJyaWRlIGNhbGwoaW5wdXRzOiBUZW5zb3J8VGVuc29yW10sIGt3YXJnczogS3dhcmdzKTogVGVuc29yW118VGVuc29yIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBpbnB1dHMgPSBnZXRFeGFjdGx5T25lVGVuc29yKGlucHV0cyk7XG4gICAgICBpZihpbnB1dHMuZHR5cGUgIT09ICdmbG9hdDMyJykge1xuICAgICAgICAgIGlucHV0cyA9IEsuY2FzdChpbnB1dHMsICdmbG9hdDMyJyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gYWRkKG11bChpbnB1dHMsIHRoaXMuc2NhbGUpLCB0aGlzLm9mZnNldCk7XG4gICAgfSk7XG4gIH1cbn1cblxuc2VyaWFsaXphdGlvbi5yZWdpc3RlckNsYXNzKFJlc2NhbGluZyk7XG4iXX0=