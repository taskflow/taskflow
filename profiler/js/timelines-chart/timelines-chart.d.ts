interface Group {
  group: string;
  data: Line[];
}

interface Line {
  label: string;
  data: Segment[];
}

interface Segment {
  timeRange: [TS, TS];
  val: Val;
}

type TS = Date | number;

type Val = number | string; // qualitative vs quantitative

type GroupLabel = {
  group: string;
  label:string;
}

type Range<DomainType> = [DomainType, DomainType];

type Formatter<ItemType> = (item: ItemType) => string;
type CompareFn<ItemType> = (a: ItemType, b: ItemType) => number;

type Scale<DomainType, RangeType> = (input: DomainType) => RangeType;

interface TimelinesChartGenericInstance<ChainableInstance> {
  (element: HTMLElement): ChainableInstance;

  width(): number;
  width(width: number): ChainableInstance;
  maxHeight(): number;
  maxHeight(height: number): ChainableInstance;
  maxLineHeight(): number;
  maxLineHeight(height: number): ChainableInstance;
  leftMargin(): number;
  leftMargin(margin: number): ChainableInstance;
  rightMargin(): number;
  rightMargin(margin: number): ChainableInstance;
  topMargin(): number;
  topMargin(margin: number): ChainableInstance;
  bottomMargin(): number;
  bottomMargin(margin: number): ChainableInstance;

  data(): Group[];
  data(data: Group[]): ChainableInstance;

  useUtc(): boolean;
  useUtc(utc: boolean): ChainableInstance;
  timeFormat(): string;
  timeFormat(format: string): ChainableInstance;
  xTickFormat(): Formatter<Date> | null;
  xTickFormat(formatter: Formatter<Date> | null): ChainableInstance;
  dateMarker(): TS | null | boolean;
  dateMarker(date: TS | null | boolean): ChainableInstance;
  minSegmentDuration(): number;
  minSegmentDuration(duration: number): ChainableInstance;

  getNLines(): number;
  getTotalNLines(): number;

  zQualitative(): boolean;
  zQualitative(isQualitative: boolean): ChainableInstance;
  zColorScale(): Scale<Val, string>;
  zColorScale(scale: Scale<Val, string>): ChainableInstance;
  zDataLabel(): string;
  zDataLabel(text: string): ChainableInstance;
  zScaleLabel(): string;
  zScaleLabel(text: string): ChainableInstance;

  sort(cmpFn: CompareFn<string>): ChainableInstance;
  sortAlpha(ascending: boolean): ChainableInstance;
  sortChrono(ascending: boolean): ChainableInstance;
  zoomX(): Range<TS | null> | null;
  zoomX(xRange: Range<TS | null> | null): ChainableInstance;
  zoomY(): Range<number | null> | null;
  zoomY(yRange: Range<number | null> | null): ChainableInstance;
  zoomYLabels(): Range<GroupLabel | null> | null;
  zoomYLabels(yLabelRange: Range<GroupLabel | null> | null): ChainableInstance;
  onZoom(cb: (zoomX: Range<TS | null> | null, zoomY: Range<number | null> | null) => void): ChainableInstance;

  enableOverview(): boolean;
  enableOverview(enable: boolean): ChainableInstance;
  overviewDomain(): Range<TS | null>;
  overviewDomain(xRange: Range<TS | null>): ChainableInstance;

  getVisibleStructure(): Group[];
  getSvg(): string;

  enableAnimations(): boolean;
  enableAnimations(animations: boolean): ChainableInstance;

  onLabelClick(cb: (label: string, group: string) => void): ChainableInstance;
  onSegmentClick(cb: (segment: {
    group: string,
    label: string,
    val: Val,
    timeRange: Range<TS>
  }) => void): ChainableInstance;

  refresh(): ChainableInstance;
}

type TimelinesChartInstance = TimelinesChartGenericInstance<TimelinesChartInstance>;

declare function TimelinesChart(): TimelinesChartInstance;

export default TimelinesChart;
export { Group, Line, Segment, TS, TimelinesChartGenericInstance, TimelinesChartInstance, Val };
